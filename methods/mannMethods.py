from method import *


class MethodMannBase(MethodBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        self.global_step = global_step  # should be TF variable
        self.total_steps = total_steps
        self.adv_steps = FLAGS.adv_steps
        self.alpha = tf.Variable(FLAGS.alpha)
        super().__init__(source_datasets, target_dataset, *args, **kwargs)
        self.loss_names += ["task", "domain"]

    def create_optimizers(self):
        opt = super().create_optimizers()
        # We need an additional optimizer for DANN
        opt["d_opt"] = self.create_optimizer(learning_rate=FLAGS.lr)
        return opt

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_domain_loss()

    def create_model(self, model_name):
        return models.MannModelBase(num_classes=self.num_classes, num_domains=self.domain_outputs,
                                    global_step=self.global_step, total_steps=self.total_steps, model_name=model_name)

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
                       domain_embedding, fe_output, which_model, training):

        nontarget = tf.where(tf.not_equal(domain_y_true, 0))
        task_y_true = tf.gather(task_y_true, nontarget)
        task_y_pred = tf.gather(task_y_pred, nontarget)
        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.compute_domain_loss(domain_embedding, domain_y_true)
        total_loss = task_loss + self.alpha * d_loss

        return [total_loss, task_loss, d_loss]

    def compute_domain_loss(self, domain_embedding, domain_y_true):
        source_num = tf.math.count_nonzero(domain_y_true, 0)
        target_num = domain_y_true.shape[0] - source_num
        d_loss = self.domain_loss(domain_embedding, source_num, target_num)
        return d_loss

    def compute_gradients(self, tape, losses, which_model):
        total_loss, task_loss, d_loss = losses
        grad = tape.gradient(total_loss,
                             self.model[which_model].trainable_variables_task_fe)
        # d_grad = self.compute_domain_gradient(tape, d_loss, which_model)
        return grad

    def compute_domain_gradient(self, tape, domain_loss, which_model):
        d_grad = tape.gradient(domain_loss,
                               self.model[which_model].trainable_variables_embedder)
        return d_grad

    def apply_gradients(self, gradients, which_model):
        grad = gradients
        self.apply_task_fe_gradients(grad, which_model)
        # self.apply_domain_gradients(d_grad, which_model)

    def apply_task_fe_gradients(self, gradient, which_model):
        self.opt[which_model]["opt"].apply_gradients(zip(gradient,
                                                         self.model[which_model].trainable_variables_task_fe))

    def apply_domain_gradients(self, gradient, which_model):
        self.opt[which_model]["d_opt"].apply_gradients(zip(gradient,
                                                           self.model[which_model].trainable_variables_embedder))

    @tf.function
    def _train_step(self, all_data_sources, all_data_target):
        """ The compiled part of train_step. We can't compile everything since
        some parts of the model need to know the shape of the data apparently.

        The first batch is passed in because to compile this, TF needs to know
        the shape. Doesn't look pretty... but it runs...
        """
        for i in range(self.ensemble_size):
            # Get random batch for this model in the ensemble (either same for
            # all or different for each)
            if FLAGS.ensemble_same_data:
                data_sources = all_data_sources[0]
                data_target = all_data_target[0]
            else:
                data_sources = all_data_sources[i]
                data_target = all_data_target[i]

            # Prepare
            x, task_y_true, domain_y_true = self.prepare_data(data_sources,
                                                              data_target)

            for j in range(FLAGS.adv_steps):
                with tf.GradientTape(persistent=True) as d_tape:
                    task_y_pred, domain_embedding, fe_output = self.call_model(
                        x, which_model=i, training=True)
                    d_loss = -self.compute_domain_loss(domain_embedding, domain_y_true)
                d_gradient = self.compute_domain_gradient(d_tape, d_loss, which_model=i)
                del d_tape
                self.apply_domain_gradients(d_gradient, which_model=i)
            # print('max-domain:%f' % ( d_loss.numpy()))
            # Run batch through the model and compute loss
            with tf.GradientTape(persistent=True) as tape:
                task_y_pred, domain_embedding, fe_output = self.call_model(
                    x, which_model=i, training=True)
                losses = self.compute_losses(x, task_y_true, domain_y_true,
                                             task_y_pred, domain_embedding, fe_output, which_model=i,
                                             training=True)
            # print('min-total:%f,task:%f,domain:%f' % (losses[0].numpy(), losses[1].numpy(), losses[2].numpy()))
            # Update model
            gradients = self.compute_gradients(tape, losses, which_model=i)
            del tape
            self.apply_gradients(gradients, which_model=i)

            
    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
                       domain_y_pred):
        """ Optionally do something with the data after feeding through the
        model. Since the model outputs logits, here we actually take the softmax
        so that during evaluation we have probability distributions. """
        task_y_pred = tf.nn.softmax(task_y_pred)
        domain_y_pred = domain_y_true
        return task_y_true, task_y_pred, domain_y_true, domain_y_pred


@register_method("mg")
class MethodMG(MethodMannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset,
                         global_step, total_steps, *args, **kwargs)

    def create_model(self, model_name):
        return models.MGModel(num_classes=self.num_classes, num_domains=self.domain_outputs,
                              global_step=self.global_step, total_steps=self.total_steps, model_name=model_name)


@register_method("smd")
class MethodMG(MethodMannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset,
                         global_step, total_steps, *args, **kwargs)

    def create_model(self, model_name):
        return models.SMDModel(num_classes=self.num_classes, num_domains=self.domain_outputs,
                              global_step=self.global_step, total_steps=self.total_steps, model_name=model_name)

@register_method("smd2")
class MethodMG(MethodMannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset,
                         global_step, total_steps, *args, **kwargs)

    def create_model(self, model_name):
        return models.SMD2Model(num_classes=self.num_classes, num_domains=self.domain_outputs,
                              global_step=self.global_step, total_steps=self.total_steps, model_name=model_name)


def make_domain_loss():
    def split_source_target(fe, s_num, t_num):
        [sx, tx] = tf.split(fe, [s_num, t_num], axis=0)
        return sx, tx

    def linear_loss(fe, s_num, t_num):  # linear mmd
        fe_source, fe_target = split_source_target(fe, s_num, t_num)
        diff = tf.math.reduce_mean(fe_source, 0, keepdims=False) - tf.math.reduce_mean(fe_target, 0, keepdims=False)
        return tf.reduce_sum(tf.multiply(diff, diff))

    return linear_loss
