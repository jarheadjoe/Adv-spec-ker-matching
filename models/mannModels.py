from model import *

# 可学习度量对齐框架
class MannModelBase(ModelBase):
    @property
    def trainable_variables_embedder(self):
        return self._get_trainable_variables(self.domain_embedder)

    @property
    def trainable_variables_task_fe_em(self):
        return self.trainable_variables_fe \
               + self.trainable_variables_task \
               + self.trainable_variables_embedder

    @property
    def trainable_variables(self):
        """ Returns all trainable variables in the model """
        return self.trainable_variables_task_fe_em

    def call_domain_embedder(self, fe, which_fe=None, which_tc=None,
                             which_dc=None, **kwargs):
        if which_dc is not None:
            assert isinstance(self.domain_embedder, list)
            return self.domain_embedder[which_dc](fe, **kwargs)

        return self.domain_embedder(fe, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        self.set_learning_phase(training)
        fe = self.call_feature_extractor(inputs, **kwargs)
        task = self.call_task_classifier(fe, **kwargs)
        em = self.call_domain_embedder(fe, **kwargs)  # maybe domain classifer/domain embedding
        return task, em, fe


class MGModel(MannModelBase):
    def __init__(self, num_classes, num_domains, model_name, global_step, total_steps, **kwargs):
        super().__init__(**kwargs)
        grl_schedule = baseModel.DannGrlSchedule(total_steps)
        self.flip_gradient = baseModel.FlipGradient(global_step, grl_schedule)
        self.num_classes = num_classes
        self.num_domains = num_domains

        fe_maker = get_model(model_name)
        self.feature_extractor = fe_maker.make_feature_extractor()
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_classes),
        ])
        self.domain_embedder = tf.keras.Sequential([
            tf.keras.layers.Dense(128,kernel_initializer='random_uniform'),
            tf.keras.layers.Activation("relu")
        ])

class SMDModel(MannModelBase):
    def __init__(self, num_classes, num_domains, model_name, global_step, total_steps, **kwargs):
        super().__init__(**kwargs)
        grl_schedule = baseModel.DannGrlSchedule(total_steps)
        self.flip_gradient = baseModel.FlipGradient(global_step, grl_schedule)
        self.num_classes = num_classes
        self.num_domains = num_domains

        fe_maker = get_model(model_name)
        self.feature_extractor = fe_maker.make_feature_extractor()
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_classes),
        ])
        self.domain_embedder = baseModel.DSKN()

class SMD2Model(MannModelBase):
    def __init__(self, num_classes, num_domains, model_name, global_step, total_steps, **kwargs):
        super().__init__(**kwargs)
        grl_schedule = baseModel.DannGrlSchedule(total_steps)
        self.flip_gradient = baseModel.FlipGradient(global_step, grl_schedule)
        self.num_classes = num_classes
        self.num_domains = num_domains

        fe_maker = get_model(model_name)
        self.feature_extractor = fe_maker.make_feature_extractor()
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_classes),
        ])
        self.domain_embedder = baseModel.DSKN2()