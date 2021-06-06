import tensorflow as tf
from functools import partial

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.math.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.math.reduce_mean(kernel(x, x))
    cost += tf.math.reduce_mean(kernel(y, y))
    cost -= 2 * tf.math.reduce_mean(kernel(x, y))
    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def KMMD(Xs,Xt):
    # sigmas=[1e-2,0.1,1,5,10,20,25,30,35,100]
    # guassian_kernel=partial(kernel,sigmas=tf.constant(sigmas))
    # cost = tf.reduce_mean(guassian_kernel(Xs, Xs))
    # cost += tf.reduce_mean(guassian_kernel(Xt, Xt))
    # cost -= 2 * tf.reduce_mean(guassian_kernel(Xs, Xt))
    # cost = tf.where(cost > 0, cost, 0)

    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    cost= maximum_mean_discrepancy(Xs, Xt, kernel=gaussian_kernel)
    return cost

def coral_loss( h_src, h_trg, gamma=1e-3):
    # regularized covariances (D-Coral is not regularized actually..)
    # First: subtract the mean from the data matrix
    source_batch_size = tf.cast(tf.shape(h_src)[0], tf.float32)
    target_batch_size = tf.cast(tf.shape(h_trg)[0], tf.float32)
    h_src = h_src - tf.math.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.math.reduce_mean(h_trg, axis=0)
    cov_source = (1. / (source_batch_size - 1)) * tf.matmul(h_src, h_src,
                                                     transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
    cov_target = (1. / (target_batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                     transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
    # cov_source=tf.linalg.cholesky(cov_source)
    # cov_target=tf.linalg.cholesky(cov_target)
    return tf.math.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))


def matchnorm(x1,x2):
    return tf.sqrt(tf.reduce_sum((x1-x2)**2))
def scm(sx1,sx2,k):
    ss1=tf.math.reduce_mean(sx1**k,axis=0)
    ss2=tf.math.reduce_mean(sx2**k,axis=0)
    return matchnorm(ss1,ss2)
def mmatch(x1,x2,n_moments):
    mx1=tf.math.reduce_mean(x1,axis=0)
    mx2=tf.math.reduce_mean(x2,axis=0)
    sx1=x1-mx1
    sx2=x2-mx2
    dm=matchnorm(mx1,mx2)
    scms=dm
    for i in range(n_moments-1):
        scms+=scm(sx1,sx2,i+2)
    return scms

def HoMM3( xs, xt):
    xs = xs - tf.math.reduce_mean(xs, axis=0)
    # xs=self.decoupling(xs)
    xt = xt - tf.reduce_mean(xt, axis=0)
    # xt=self.decoupling(xt)
    xs=tf.expand_dims(xs,axis=-1)
    xs = tf.expand_dims(xs, axis=-1)
    xt = tf.expand_dims(xt, axis=-1)
    xt = tf.expand_dims(xt, axis=-1)
    xs_1=tf.transpose(xs,[0,2,1,3])
    xs_2 = tf.transpose(xs, [0, 2, 3, 1])
    xt_1 = tf.transpose(xt, [0, 2, 1, 3])
    xt_2 = tf.transpose(xt, [0, 2, 3, 1])
    HR_Xs=xs*xs_1*xs_2
    HR_Xs=tf.reduce_mean(HR_Xs,axis=0)
    HR_Xt = xt * xt_1 * xt_2
    HR_Xt = tf.math.reduce_mean(HR_Xt, axis=0)
    return tf.math.reduce_mean(tf.square(tf.subtract(HR_Xs, HR_Xt)))