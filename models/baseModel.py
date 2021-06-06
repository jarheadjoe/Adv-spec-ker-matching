import tensorflow as tf
import tensorflow.keras.layers as klayer
import tensorflow.keras.backend as K


class Fcn(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Fcn, self).__init__(**kwargs)
        self.fcn = tf.keras.Sequential([
            klayer.Conv1D(filters=128, kernel_size=8, padding="same",
                          use_bias=False),
            klayer.BatchNormalization(),
            klayer.Activation("relu"),

            klayer.Conv1D(filters=256, kernel_size=5, padding="same",
                          use_bias=False),
            klayer.BatchNormalization(),
            klayer.Activation("relu"),

            klayer.Conv1D(filters=128, kernel_size=3, padding="same",
                          use_bias=False),
            klayer.BatchNormalization(),
            klayer.Activation("relu"),

            klayer.GlobalAveragePooling1D(),
        ])

    def call(self, inputs, **kwargs):
        return self.fcn(inputs)


class LstmFcn(tf.keras.Model):
    def __init__(self, units=64, **kwargs):
        super(LstmFcn, self).__init__()
        self.units = units
        self.fcn = Fcn()
        self.lstm = klayer.LSTM(self.units)

    def call(self, inputs, **kwargs):
        lstm_out = self.lstm(inputs)
        lstm_out = klayer.Dropout(0.8)(lstm_out)
        fcn_out = self.fcn(inputs)
        return klayer.concatenate([lstm_out, fcn_out])


@tf.custom_gradient
def flip_gradient(x, grl_lambda):
    """ Forward pass identity, backward pass negate gradient and multiply by  """
    grl_lambda = tf.cast(grl_lambda, dtype=tf.float32)

    def grad(dy):
        # the 0 is for grl_lambda, which doesn't have a gradient
        return tf.negative(dy) * grl_lambda * tf.ones_like(x), 0

    return x, grad


class FlipGradient(klayer.Layer):
    """
    Gradient reversal layer

    global_step = tf.Variable storing the current step
    schedule = a function taking the global_step and computing the grl_lambda,
        e.g. `lambda step: 1.0` or some more complex function.
    """

    def __init__(self, global_step, grl_schedule, **kwargs):
        super().__init__(**kwargs)
        self.global_step = global_step
        self.grl_schedule = grl_schedule

    def call(self, inputs, **kwargs):
        """ Calculate grl_lambda first based on the current global step (a
        variable) and then create the layer that does nothing except flip
        the gradients """
        grl_lambda = self.grl_schedule(self.global_step)
        return flip_gradient(inputs, grl_lambda)


def DannGrlSchedule(num_steps):
    """ GRL schedule from DANN paper """
    num_steps = tf.cast(num_steps, tf.float32)

    def schedule(step):
        step = tf.cast(step, tf.float32)
        return 2 / (1 + tf.exp(-10 * (step / (num_steps + 1)))) - 1

    return schedule


def CosActivation(x):
    return K.cos(x)


class SpectralMap(tf.keras.Model):
    def __init__(self, units,**kwargs):
        super(SpectralMap, self).__init__()
        self.units = units

        self.layer = klayer.Dense(self.units,kernel_initializer='random_uniform')
        self.norm = klayer.BatchNormalization()
        self.act = klayer.Activation(activation=CosActivation)

    def call(self, inputs,**kwargs):
        layer_out = self.layer(inputs)
        out_norm = self.norm(layer_out)
        return self.act(out_norm)

class SpectralMap2(tf.keras.Model):
    def __init__(self, units,**kwargs):
        super(SpectralMap2, self).__init__()
        self.units = units

        self.layer1 = klayer.Dense(self.units/2,kernel_initializer='random_uniform')
        self.layer2 = klayer.Dense(self.units/2,kernel_initializer='random_uniform')
        self.norm = klayer.BatchNormalization()
        self.act1 = klayer.Activation(activation=CosActivation)
        self.act2 = klayer.Activation("relu")

    def call(self, inputs,**kwargs):
        layer_out1 = self.layer1(inputs)
        layer_out2 = self.layer2(inputs)
        out_norm1 = self.norm(layer_out1)
        out_norm2 = self.norm(layer_out2)
        out1 = self.act1(out_norm1)
        out2 = self.act2(out_norm2)
        return tf.concat([out1,out2],axis=1)


class DSKN(tf.keras.Model):
    def __init__(self,**kwargs):
        super(DSKN, self).__init__()
        self.layer1 = SpectralMap(units=256)
        self.layer2 = SpectralMap(units=128)

    def call(self, inputs,**kwargs):
        layer1_out = self.layer1(inputs)
        layer2_out = self.layer2(layer1_out)
        return layer2_out


class DSKN2(tf.keras.Model):
    def __init__(self,**kwargs):
        super(DSKN2, self).__init__()
        self.layer1 = SpectralMap2(units=256)
        self.layer2 = SpectralMap2(units=128)

    def call(self, inputs,**kwargs):
        layer1_out = self.layer1(inputs)
        layer2_out = self.layer2(layer1_out)
        return layer2_out


# inputs=tf.ones([10,3,8])
# fcn=Fcn()
# outputs=fcn(inputs)
# print()