from method import *


class MethodDannBase(MethodBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        self.global_step = global_step  # should be TF variable
        self.total_steps = total_steps
        super().__init__(source_datasets, target_dataset, *args, **kwargs)
        self.loss_names += ["task", "domain"]

    def create_optimizers(self):
        opt = super().create_optimizers()
        # We need an additional optimizer for DANN
        opt["d_opt"] = self.create_optimizer(
            learning_rate=FLAGS.lr * FLAGS.lr_domain_mult)
        return opt

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_loss()

    def prepare_data(self, data_sources, data_target):
        assert data_target is not None, "cannot run DANN without target"
        x_a, y_a, domain_a = data_sources
        x_b, y_b, domain_b = data_target

        # Concatenate all source domains' data
        x_a = tf.concat(x_a, axis=0)
        y_a = tf.concat(y_a, axis=0)
        domain_a = tf.concat(domain_a, axis=0)

        # Concatenate for adaptation - concatenate source labels with all-zero
        # labels for target since we can't use the target labels during
        # unsupervised domain adaptation
        x = tf.concat((x_a, x_b), axis=0)
        task_y_true = tf.concat((y_a, tf.zeros_like(y_b)), axis=0)
        domain_y_true = tf.concat((domain_a, domain_b), axis=0)

        return x, task_y_true, domain_y_true

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
                       domain_y_pred, fe_output, which_model, training):
        nontarget = tf.where(tf.not_equal(domain_y_true, 0))
        task_y_true = tf.gather(task_y_true, nontarget)
        task_y_pred = tf.gather(task_y_pred, nontarget)

        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]

    def compute_gradients(self, tape, losses, which_model):
        total_loss, task_loss, d_loss = losses
        grad = tape.gradient(total_loss,
                             self.model[which_model].trainable_variables_task_fe_domain)
        d_grad = tape.gradient(d_loss,
                               self.model[which_model].trainable_variables_domain)
        return [grad, d_grad]

    def apply_gradients(self, gradients, which_model):
        grad, d_grad = gradients
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
                                                         self.model[which_model].trainable_variables_task_fe_domain))
        # Update discriminator again
        self.opt[which_model]["d_opt"].apply_gradients(zip(d_grad,
                                                           self.model[which_model].trainable_variables_domain))

    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
                       domain_y_pred):
        """ Optionally do something with the data after feeding through the
        model. Since the model outputs logits, here we actually take the softmax
        so that during evaluation we have probability distributions. """
        task_y_pred = tf.nn.softmax(task_y_pred)
        domain_y_pred = tf.nn.softmax(domain_y_pred)
        return task_y_true, task_y_pred, domain_y_true, domain_y_pred


@register_method("dann")
class MethodDann(MethodDannBase):
    def create_model(self, model_name):
        return models.DannModel(num_classes=self.num_classes, num_domains=self.domain_outputs,
                                global_step=self.global_step, total_steps=self.total_steps, model_name=model_name)


@register_method("codats")
class MethodCodats(MethodDannBase):
    def create_model(self, model_name):
        return models.CodatsModel(num_classes=self.num_classes, num_domains=self.domain_outputs,
                                  global_step=self.global_step, total_steps=self.total_steps, model_name=model_name)
