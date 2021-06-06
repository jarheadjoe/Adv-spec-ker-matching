import tensorflow as tf
from models import baseModel

models = {}


def register_model(name):
    """ Add model to the list of models, e.g. add @register_model("name")
    before a class definition """
    assert name not in models, "duplicate model named " + name

    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def get_model(name, *args, **kwargs):
    """ Based on the given name, call the correct model """
    assert name in models.keys(), \
        "Unknown model name " + name
    return models[name](*args, **kwargs)


def list_models():
    """ Returns list of all the available models """
    return list(models.keys())


class ModelBase(tf.keras.Model):
    """ Base model class (inheriting from Keras' Model class) """

    def __init__(self, *args, **kwargs):  # if_domain for dann or dskn
        super().__init__(*args, **kwargs)

    def _get_trainable_variables_list(self, model_list):
        """ Get all trainable variables if model is a list """
        model_vars = []

        for m in model_list:
            model_vars += m.trainable_variables

        return model_vars

    def _get_trainable_variables(self, model):
        """ Get trainable variables if model is a list or not """
        if isinstance(model, list):
            return self._get_trainable_variables_list(model)

        return model.trainable_variables

    @property
    def trainable_variables_fe(self):
        return self._get_trainable_variables(self.feature_extractor)

    @property
    def trainable_variables_task(self):
        return self._get_trainable_variables(self.task_classifier)

    @property
    def trainable_variables_task_fe(self):
        return self.trainable_variables_fe \
               + self.trainable_variables_task

    @property
    def trainable_variables(self):
        """ Returns all trainable variables in the model """
        return self.trainable_variables_task_fe

    def set_learning_phase(self, training):
        # Manually set the learning phase since we probably aren't using .fit()
        # but layers like batch norm and dropout still need to know if
        # training/testing
        if training is True:
            tf.keras.backend.set_learning_phase(1)
        elif training is False:
            tf.keras.backend.set_learning_phase(0)

    # Allow easily overriding each part of the call() function, without having
    # to override call() in its entirety
    def call_feature_extractor(self, inputs, which_fe=None, which_tc=None,
                               which_dc=None, **kwargs):
        if which_fe is not None:
            assert isinstance(self.feature_extractor, list)
            return self.feature_extractor[which_fe](inputs, **kwargs)

        return self.feature_extractor(inputs, **kwargs)

    def call_task_classifier(self, fe, which_fe=None, which_tc=None,
                             which_dc=None, **kwargs):
        if which_tc is not None:
            assert isinstance(self.task_classifier, list)
            return self.task_classifier[which_tc](fe, **kwargs)

        return self.task_classifier(fe, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        self.set_learning_phase(training)
        fe = self.call_feature_extractor(inputs, **kwargs)
        task = self.call_task_classifier(fe, **kwargs)
        domain = tf.zeros([task.shape[0],2],tf.int32)
        return task, domain, fe



class FeatureExtractorMaker():
    def __init__(self, **kwargs):
        pass

    def make_feature_extractor(self, **kwargs):
        raise NotImplementedError("must implement for ModelMaker class")


@register_model("fcn")
class FcnFeatureExtractorMaker(FeatureExtractorMaker):
    def make_feature_extractor(self, **kwargs):
        return baseModel.Fcn()


@register_model("lstmfcn")
class LstmfcnFeatureExtractorMaker(FeatureExtractorMaker):
    def make_feature_extractor(self, **kwargs):
        return baseModel.LstmFcn()


class BasicModel(ModelBase):
    def __init__(self, num_classes, num_domains, model_name, **kwargs):
        super().__init__( **kwargs)
        self.num_classes = num_classes
        self.num_domains = num_domains

        fe_maker = get_model(model_name)
        self.feature_extractor = fe_maker.make_feature_extractor()
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_classes),
        ])



