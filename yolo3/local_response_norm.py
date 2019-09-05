from keras.layers.core import Layer
from keras import backend as K

"""
LocalResponseNorm Layer Definition in Keras

:param float alpha:
:param int k:
:param float beta:
:param int n:
:param **kwargs of Layer Class from Keras:

This code is adapted from pylearn2.
License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
"""

class LocalResponseNorm(Layer):

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LocalResponseNorm, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2
        input_sqr = K.square(x)

        extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
        input_sqr = K.concatenate(
                                [
                                    extra_channels[:, :half_n, :, :],
                                    input_sqr,
                                    extra_channels[:, half_n + int(ch):, :, :]
                                ],
                                axis=1
                                )

        scale = self.k  # offset for the scale
        norm_alpha = self.alpha / self.n  # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + int(ch), :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LocalResponseNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""
LocalResponseNorm2D Layer Definition in Keras

:param floatalpha:
:param int k:
:param float beta:
:param int n:
:param **kwargs of Layer Class from Keras:

This code is adapted from pylearn2.
License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
"""
class LocalResponseNorm2D(LocalResponseNorm):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError(
                                    """LocalResponseNorm2D only works
                                    with odd n. n provided: """ + str(n))
        super(LocalResponseNorm2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LocalResponseNorm2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, 1:, 1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
