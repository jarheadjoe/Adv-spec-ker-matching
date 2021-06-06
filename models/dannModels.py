from model import *
# dann 框架
class DannModelBase(ModelBase):
    @property
    def trainable_variables_domain(self):
        return self._get_trainable_variables(self.domain_classifier)

    @property
    def trainable_variables_task_fe_domain(self):
        return self.trainable_variables_fe \
               + self.trainable_variables_task \
               + self.trainable_variables_domain

    @property
    def trainable_variables(self):
        """ Returns all trainable variables in the model """
        return self.trainable_variables_task_fe_domain

    def call_domain_classifier(self, fe, task, which_fe=None, which_tc=None,
                               which_dc=None, **kwargs):
        if which_dc is not None:
            assert isinstance(self.domain_classifier, list)
            return self.domain_classifier[which_dc](fe, **kwargs)

        return self.domain_classifier(fe, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        self.set_learning_phase(training)
        fe = self.call_feature_extractor(inputs, **kwargs)
        task = self.call_task_classifier(fe, **kwargs)
        domain = self.call_domain_classifier(fe, task, **kwargs)  # maybe domain classifer/domain embedding
        return task, domain, fe

class DannModel(DannModelBase):
    def __init__(self, num_classes, num_domains, model_name, global_step, total_steps, **kwargs):
        super().__init__( **kwargs)
        grl_schedule = baseModel.DannGrlSchedule(total_steps)
        self.flip_gradient = baseModel.FlipGradient(global_step, grl_schedule)
        self.num_classes = num_classes
        self.num_domains = num_domains

        fe_maker = get_model(model_name)
        self.feature_extractor = fe_maker.make_feature_extractor()
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(num_classes),
        ])
        self.domain_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(num_domains),
        ])

    def call_domain_classifier(self, fe, task, **kwargs):
        # Pass FE output through GRL then to DC
        grl_output = self.flip_gradient(fe, **kwargs)
        return super().call_domain_classifier(grl_output, task, **kwargs)

class CodatsModel(DannModelBase):
    def __init__(self, num_classes, num_domains, model_name, global_step, total_steps, **kwargs):
        super().__init__( **kwargs)
        grl_schedule = baseModel.DannGrlSchedule(total_steps)
        self.flip_gradient = baseModel.FlipGradient(global_step, grl_schedule)
        self.num_classes = num_classes
        self.num_domains = num_domains

        fe_maker = get_model(model_name)
        self.feature_extractor = fe_maker.make_feature_extractor()
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_classes),
        ])
        self.domain_classifier = tf.keras.Sequential([
            # Note: alternative is Dense(128, activation="tanh") like used by
            # https://arxiv.org/pdf/1902.09820.pdf They say dropout of 0.7 but
            # I'm not sure if that means 1-0.7 = 0.3 or 0.7 itself.
            tf.keras.layers.Dense(500, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(500, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(self.num_domains),
        ])

    def call_domain_classifier(self, fe, task, **kwargs):
        # Pass FE output through GRL then to DC
        grl_output = self.flip_gradient(fe, **kwargs)
        return super().call_domain_classifier(grl_output, task, **kwargs)