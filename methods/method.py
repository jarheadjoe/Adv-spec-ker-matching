"""
Methods
"""
import tensorflow as tf
import utils
# import tensorflow_addons as tfa

from absl import flags

import models
import load_datasets

FLAGS = flags.FLAGS

flags.DEFINE_float("lr", 0.0001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")
flags.DEFINE_float("hda_l2", 1000.0, "Weight for regularizing each domain's feature extractor weights to be similar")
flags.DEFINE_boolean("hda_by_layer", False,
                     "Regularize lower layers less and higher layers more, only matters if hda_l2 != 0")
flags.DEFINE_boolean("ensemble_same_data", False,
                     "Train each model on the same batch of data, or if false use a different random batch for each model")

flags.DEFINE_float("alpha", 1.0, "factor for domain loss")
flags.DEFINE_integer("adv_steps", 1, "adv_step for mann methods")
methods = {}


def register_method(name):
    """ Add method to the list of methods, e.g. add @register_method("name")
    before a class definition """
    assert name not in methods, "duplicate method named " + name

    def decorator(cls):
        methods[name] = cls
        return cls

    return decorator


def get_method(name, *args, **kwargs):
    """ Based on the given name, call the correct method """
    assert name in methods.keys(), \
        "Unknown method name " + name
    return methods[name](*args, **kwargs)


def list_methods():
    """ Returns list of all the available methods """
    return list(methods.keys())


class MethodBase:
    def __init__(self, source_datasets, target_dataset, model_name,
                 *args, ensemble_size=1, trainable=True, moving_average=False,
                 share_most_weights=False, **kwargs):
        self.source_datasets = source_datasets
        self.target_dataset = target_dataset
        self.moving_average = moving_average
        self.ensemble_size = ensemble_size
        assert ensemble_size > 0, "ensemble_size should be >= 1"
        self.share_most_weights = share_most_weights  # for HeterogeneousBase

        # Support multiple targets when we add that functionality
        self.num_source_domains = len(source_datasets)
        self.num_domains = len(source_datasets)

        if target_dataset is not None:
            if isinstance(target_dataset, list):
                self.num_domains += len(target_dataset)
            elif isinstance(target_dataset, load_datasets.Dataset):
                self.num_domains += 1
            else:
                raise NotImplementedError("target_dataset should be either one "
                                          "load_datasets.Dataset() or a list of them, "
                                          "but is " + str(target_dataset))

        # How to calculate the number of domain outputs
        self.domain_outputs = self.calculate_domain_outputs()

        # We need to know the num_classes for creating the model
        # We'll just pick the first source since we have to have at least one
        # source and we've already verified they're all the same in load_da()
        self.num_classes = source_datasets[0].num_classes

        # What we want in the checkpoint
        self.checkpoint_variables = {}

        # Initialize components -- support ensemble, training all simultaneously
        # I think will be faster / more efficient overall time-wise
        self.create_iterators()
        self.opt = [self.create_optimizers() for _ in range(ensemble_size)]
        self.model = [self.create_model(model_name) for _ in range(ensemble_size)]
        self.create_losses()

        # Checkpoint/save the model and optimizers
        for i, model in enumerate(self.model):
            self.checkpoint_variables["model_" + str(i)] = model

        for i, opt_dict in enumerate(self.opt):
            for name, opt in opt_dict.items():
                self.checkpoint_variables["opt_" + name + "_" + str(i)] = opt

        # Names of the losses returned in compute_losses
        self.loss_names = ["total"]

        # Should this method be trained (if not, then in main.py the config
        # is written and then it exits)
        self.trainable = trainable

    def calculate_domain_outputs(self):
        """ Calculate the number of outputs for the domain classifier. By
        default it's the number of domains. However, for example, in domain
        generalization we ignore the target, so it'll actually be the number of
        source domains only, in which case override this function. """
        return self.num_domains

    def create_iterators(self):
        """ Get the source/target train/eval datasets """
        self.source_train_iterators = [iter(x.train) for x in self.source_datasets]
        self.source_train_eval_datasets = [x.train_evaluation for x in self.source_datasets]
        self.source_test_eval_datasets = [x.test_evaluation for x in self.source_datasets]

        if self.target_dataset is not None:
            self.target_train_iterator = iter(self.target_dataset.train)
            self.target_train_eval_dataset = self.target_dataset.train_evaluation
            self.target_test_eval_dataset = self.target_dataset.test_evaluation
        else:
            self.target_train_iterator = None
            self.target_train_eval_dataset = None
            self.target_test_eval_dataset = None

    def create_optimizer(self, *args, **kwargs):
        """ Create a single optimizer """
        opt = tf.keras.optimizers.Adam(*args, **kwargs)

        # if self.moving_average:
        #     opt = tfa.optimizers.MovingAverage(opt)

        return opt

    def create_optimizers(self):
        return {"opt": self.create_optimizer(learning_rate=FLAGS.lr)}

    def create_model(self, model_name):
        return models.BasicModel(self.num_classes, self.domain_outputs,
                                 model_name=model_name)

    def create_losses(self):
        self.task_loss = make_loss()

    def get_next_train_data(self):
        """ Get next batch of training data """
        # Note we will use this same exact data in Metrics() as we use in
        # train_step()
        data_sources = [next(x) for x in self.source_train_iterators]
        data_target = next(self.target_train_iterator) \
            if self.target_train_iterator is not None else None
        return self.get_next_batch_both(data_sources, data_target)

    def domain_label(self, index, is_target):
        """ Default domain labeling. Indexes should be in [0,+inf) and integers.
        0 = target
        1 = source #0
        2 = source #1
        3 = source #2
        ...
        """
        if is_target:
            return 0
        else:
            return index + 1

    @tf.function
    def get_next_batch_both(self, data_sources, data_target):
        """ Compile for training. Don't for evaluation (called directly,
        not this _both function). """
        data_sources = self.get_next_batch_multiple(data_sources, is_target=False)
        data_target = self.get_next_batch_single(data_target, is_target=True)
        return data_sources, data_target

    def get_next_batch_multiple(self, data, is_target):
        """
        Get next set of training data. data should be a list of data (probably
        something like [next(x) for x in iterators]).

        Returns: (
            [x_a1, x_a2, x_a3, ...],
            [y_a1, y_a2, y_a3, ...],
            [domain_a1, domain_a2, domain_a3, ...]
        )
        """
        if data is None:
            return None

        assert not is_target or len(data) == 1, \
            "only support one target at present"

        xs = []
        ys = []
        ds = []

        for i, (x, y) in enumerate(data):
            xs.append(x)
            ys.append(y)
            ds.append(tf.ones_like(y) * self.domain_label(index=i,
                                                          is_target=is_target))

        return (xs, ys, ds)

    def get_next_batch_single(self, data, is_target, index=0):
        """
        Get next set of training data. data should be a single batch (probably
        something like next(iterator)). When processing target data, index
        must be 0 since we only support one target at the moment. However,
        during evaluation we evaluate each source's data individually so if
        is_target is False, then index can be whichever source domain was
        passed.

        Returns: (x, y, domain)
        """
        if data is None:
            return None

        assert not is_target or index == 0, \
            "only support one target at present"

        x, y = data
        d = tf.ones_like(y) * self.domain_label(index=index, is_target=is_target)
        data_target = (x, y, d)

        return data_target

    # Allow easily overriding each part of the train_step() function, without
    # having to override train_step() in its entirety
    # def prepare_data(self, data_sources, data_target):
    #     """ Prepare the data for the model, e.g. by concatenating all sources
    #     together. Note: do not put code in here that changes the domain labels
    #     since you presumably want that during evaluation too. Put that in
    #     domain_label() """
    #     # By default (e.g. for no adaptation or domain generalization), ignore
    #     # the target data
    #     x_a, y_a, domain_a = data_sources
    #     x = tf.concat(x_a, axis=0)
    #     task_y_true = tf.concat(y_a, axis=0)
    #     domain_y_true = tf.concat(domain_a, axis=0)
    #     return x, task_y_true, domain_y_true
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

    def prepare_data_eval(self, data, is_target=None):
        """ Prepare the data for the model, e.g. by concatenating all sources
        together. This is like prepare_data() but use during evaluation. """
        x, y, domain = data

        assert isinstance(x, list), \
            "Must pass x=[...] even if only one domain for tf.function consistency"
        assert isinstance(y, list), \
            "Must pass y=[...] even if only one domain for tf.function consistency"
        assert isinstance(domain, list), \
            "Must pass domain=[...] even if only one domain for tf.function consistency"

        # Concatenate all the data (e.g. if multiple source domains)
        x = tf.concat(x, axis=0)
        y = tf.concat(y, axis=0)
        domain = tf.concat(domain, axis=0)

        return x, y, domain

    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
                       domain_y_pred):
        """ Optionally do something with the data after feeding through the
        model. Since the model outputs logits, here we actually take the softmax
        so that during evaluation we have probability distributions. """
        task_y_pred = tf.nn.softmax(task_y_pred)
        domain_y_pred = domain_y_true
        return task_y_true, task_y_pred, domain_y_true, domain_y_pred

    def call_model(self, x, which_model, is_target=None, **kwargs):
        return self.model[which_model](x, **kwargs)

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
                       domain_y_pred, fe_output, which_model, training):
        # Maybe: regularization = sum(model.losses) and add to loss
        return self.task_loss(task_y_true, task_y_pred)

    def compute_gradients(self, tape, loss, which_model):
        return tape.gradient(loss,
                             self.model[which_model].trainable_variables_task_fe)

    def apply_gradients(self, grad, which_model):
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
                                                         self.model[which_model].trainable_variables_task_fe))

    def train_step(self):
        """
        Get batch of data, prepare data, run through model, compute losses,
        apply the gradients

        Override the individual parts with prepare_data(), call_model(),
        compute_losses(), compute_gradients(), and apply_gradients()

        We return the batch of data so we can use the exact same training batch
        for the "train" evaluation metrics.
        """
        # TensorFlow errors constructing the graph (with tf.function, which
        # makes training faster) if we don't know the data size. Thus, first
        # load batches, then pass to compiled train step.
        all_data_sources = []
        all_data_target = []

        for i in range(self.ensemble_size):
            data_sources, data_target = self.get_next_train_data()
            all_data_sources.append(data_sources)
            all_data_target.append(data_target)

            # If desired, use the same batch for each of the models.
            if FLAGS.ensemble_same_data:
                break

        self._train_step(all_data_sources, all_data_target)

        # We return the first one since we don't really care about the "train"
        # evaluation metrics that much.
        return all_data_sources[0], all_data_target[0]

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

            # Run batch through the model and compute loss
            with tf.GradientTape(persistent=True) as tape:
                task_y_pred, domain_y_pred, fe_output = self.call_model(
                    x, which_model=i, training=True)
                losses = self.compute_losses(x, task_y_true, domain_y_true,
                                             task_y_pred, domain_y_pred, fe_output, which_model=i,
                                             training=True)
            # print('min-total:%f,task:%f,domain:%f' % (losses[0].numpy(), losses[1].numpy(), losses[2].numpy()))
            # Update model
            gradients = self.compute_gradients(tape, losses, which_model=i)
            del tape
            self.apply_gradients(gradients, which_model=i)

    def eval_step(self, data, is_target):
        """ Evaluate a batch of source or target data, called in metrics.py.
        This preprocesses the data to have x, y, domain always be lists so
        we can use the same compiled tf.function code in eval_step_list() for
        both sources and target domains. """
        x, y, domain = data

        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        if not isinstance(domain, list):
            domain = [domain]

        return self.eval_step_list((x, y, domain), is_target)

    def tsne_step(self,data,is_target):
        x, y, domain = data

        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        if not isinstance(domain, list):
            domain = [domain]

        return self.tsne_step_list((x, y, domain), is_target)


    def add_multiple_losses(self, losses, average=False):
        """
        losses = [
            [total_loss1, task_loss1, ...],
            [total_loss2, task_loss2, ...],
            ...
        ]

        returns [total_loss, task_loss, ...] either the sum or average
        """
        losses_added = None

        for loss_list in losses:
            # If no losses yet, then just set to this
            if losses_added is None:
                losses_added = loss_list
            # Otherwise, add to the previous loss values
            else:
                assert len(losses_added) == len(loss_list), \
                    "subsequent losses have different length than the first"

                for i, loss in enumerate(loss_list):
                    losses_added[i] += loss

        assert losses_added is not None, \
            "must return losses from at least one domain"

        if average:
            averaged_losses = []

            for loss in losses_added:
                averaged_losses.append(loss / len(losses))

            return averaged_losses
        else:
            return losses_added

    # @tf.function  # faster not to compile
    def eval_step_list(self, data, is_target):
        """ Override preparation in prepare_data_eval() """
        x, orig_task_y_true, orig_domain_y_true = self.prepare_data_eval(data,
                                                                         is_target)

        task_y_true_list = []
        task_y_pred_list = []
        domain_y_true_list = []
        domain_y_pred_list = []
        # losses_list = []

        for i in range(self.ensemble_size):
            # Run through model
            task_y_pred, domain_y_pred, fe_output = self.call_model(x,
                                                                    which_model=i, is_target=is_target, training=False)

            # Calculate losses
            # losses = self.compute_losses(x, orig_task_y_true,
            #                              orig_domain_y_true, task_y_pred, domain_y_pred, fe_output,
            #                              which_model=i, training=False)
            # # if not isinstance(losses, list):
            # #     losses = [losses]
            # #
            # # losses_list.append(losses)
            #
            # if not isinstance(losses, list):
            #     losses = [losses]
            # else:
            #     losses = [losses[1]]
            #
            # losses_list.append(losses)

            # Post-process data (e.g. compute softmax from logits)
            task_y_true, task_y_pred, domain_y_true, domain_y_pred = \
                self.post_data_eval(orig_task_y_true, task_y_pred,
                                    orig_domain_y_true, domain_y_pred)

            task_y_true_list.append(task_y_true)
            task_y_pred_list.append(task_y_pred)
            domain_y_true_list.append(domain_y_true)
            domain_y_pred_list.append(domain_y_pred)

        # Combine information from each model in the ensemble -- averaging.
        #
        # Note: this is how the ensemble predictions are made with InceptionTime
        # having an ensemble of 5 models -- they average the softmax outputs
        # over the ensemble (and we now have softmax after the post_data_eval()
        # call). See their code:
        # https://github.com/hfawaz/InceptionTime/blob/master/classifiers/nne.py
        task_y_true_avg = tf.math.reduce_mean(task_y_true_list, axis=0)
        task_y_pred_avg = tf.math.reduce_mean(task_y_pred_list, axis=0)
        domain_y_true_avg = tf.math.reduce_mean(domain_y_true_list, axis=0)
        domain_y_pred_avg = tf.math.reduce_mean(domain_y_pred_list, axis=0)
        # losses_avg = self.add_multiple_losses(losses_list, average=True)

        # return task_y_true_avg, task_y_pred_avg, domain_y_true_avg, \
        #        domain_y_pred_avg, losses_avg
        return task_y_true_avg, task_y_pred_avg, domain_y_true_avg, \
               domain_y_pred_avg

    def tsne_step_list(self, data, is_target):
        x, orig_task_y_true, orig_domain_y_true = self.prepare_data_eval(data,
                                                                         is_target)
        # Run through model
        task_y_pred, domain_y_pred, fe_output = self.call_model(x,
                                                                which_model=0, is_target=is_target, training=False)
        return fe_output, orig_task_y_true


def make_loss(from_logits=True):
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    def loss(y_true, y_pred):
        return cce(y_true, y_pred)

    return loss


@register_method("none")
class MethodNone(MethodBase):
    def __init__(self, source_datasets, target_dataset,
                 global_step, total_steps, *args, **kwargs):
        self.global_step = global_step  # should be TF variable
        self.total_steps = total_steps
        super().__init__(source_datasets, target_dataset, *args, **kwargs)
        self.loss_names += ["task", "domain"]
