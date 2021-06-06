from method import *


class MethodNannBase(MethodBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        self.global_step = global_step  # should be TF variable
        self.total_steps = total_steps
        self.alpha = tf.Variable(FLAGS.alpha)
        super().__init__(source_datasets, target_dataset, *args, **kwargs)
        self.loss_names += ["task", "domain"]

    def create_model(self, model_name):
        return models.NannModel(num_classes=self.num_classes, num_domains=self.domain_outputs,
                                global_step=self.global_step, total_steps=self.total_steps, model_name=model_name)

    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
                       domain_y_pred):
        """ Optionally do something with the data after feeding through the
        model. Since the model outputs logits, here we actually take the softmax
        so that during evaluation we have probability distributions. """
        task_y_pred = tf.nn.softmax(task_y_pred)
        domain_y_pred = domain_y_true
        return task_y_true, task_y_pred, domain_y_true, domain_y_pred

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
                       domain_y_pred, fe_output, which_model, training):
        source_num = tf.math.count_nonzero(domain_y_true, 0)
        target_num = domain_y_true.shape[0] - source_num
        nontarget = tf.where(tf.not_equal(domain_y_true, 0))
        task_y_true = tf.gather(task_y_true, nontarget)
        task_y_pred = tf.gather(task_y_pred, nontarget)

        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(fe_output, source_num, target_num)
        total_loss = task_loss + self.alpha * d_loss

        return [total_loss, task_loss, d_loss]

    def compute_gradients(self, tape, losses, which_model):
        total_loss, task_loss, d_loss = losses
        grad = tape.gradient(total_loss,
                             self.model[which_model].trainable_variables_task_fe)

        return grad

    def apply_gradients(self, gradients, which_model):
        grad = gradients
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
                                                         self.model[which_model].trainable_variables_task_fe))


@register_method("ddc")
class MethodDcc(MethodNannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset, global_step, total_steps, *args, **kwargs)

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_mmd_loss("ddc")


@register_method("dan")
class MethodDan(MethodNannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset, global_step, total_steps, *args, **kwargs)

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_mmd_loss("dan")


@register_method("coral")
class MethodCoral(MethodNannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset, global_step, total_steps, *args, **kwargs)

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_mmd_loss("coral")


@register_method("cmd")
class MethodCmd(MethodNannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset, global_step, total_steps, *args, **kwargs)

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_mmd_loss("cmd")


@register_method("homm")
class MethodHomm(MethodNannBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        super().__init__(source_datasets, target_dataset, global_step, total_steps, *args, **kwargs)

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_mmd_loss("homm")


def make_mmd_loss(method):
    def split_source_target(fe, s_num, t_num):
        [sx, tx] = tf.split(fe, [s_num, t_num], axis=0)
        return sx, tx

    def ddc_loss(fe, s_num, t_num):  # linear mmd
        fe_source, fe_target = split_source_target(fe, s_num, t_num)
        diff = tf.math.reduce_mean(fe_source, 0, keepdims=False) - tf.math.reduce_mean(fe_target, 0, keepdims=False)
        return tf.reduce_sum(tf.multiply(diff, diff))

    def dan_loss(fe, s_num, t_num):
        fe_source, fe_target = split_source_target(fe, s_num, t_num)
        return tf.maximum(0.0001, utils.KMMD(fe_source, fe_target))

    def coral_loss(fe, s_num, t_num):
        fe_source, fe_target = split_source_target(fe, s_num, t_num)
        return utils.coral_loss(fe_source, fe_target)

    def cmd_loss(fe, s_num, t_num):
        fe_source, fe_target = split_source_target(fe, s_num, t_num)
        return utils.mmatch(fe_source, fe_target, 3)

    def homm_loss(fe, s_num, t_num):
        fe_source, fe_target = split_source_target(fe, s_num, t_num)
        return utils.HoMM3(fe_source, fe_target)

    if method == "ddc":
        return ddc_loss
    elif method == "dan":
        return dan_loss
    elif method == "coral":
        return coral_loss
    elif method == "cmd":
        return cmd_loss
    elif method == "homm":
        return homm_loss
