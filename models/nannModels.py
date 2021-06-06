from model import *
# 普通度量学习
class NannModel(ModelBase):
    def __init__(self, num_classes, num_domains, model_name, global_step, total_steps, **kwargs):
        super().__init__( **kwargs)
        self.num_classes = num_classes
        self.num_domains = num_domains
        fe_maker = get_model(model_name)
        self.feature_extractor = fe_maker.make_feature_extractor()
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_classes),
        ])