import time
import tensorflow as tf
# import tensorflow_addons as tfa

from absl import flags

from plots import generate_plots

FLAGS = flags.FLAGS


class Metrics:
    def __init__(self, log_dir, method, source_datasets, target_dataset,
                 target_domain=True):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.method = method
        self.source_datasets = source_datasets
        self.target_dataset = target_dataset
        self.num_classes = source_datasets[0].num_classes
        self.domain_outputs = method.domain_outputs  # for dann methods
        self.datasets = ["training", "validation"]
        self.target_domain = target_domain  # whether we have just source or both
        self.domains = ["source", "target"]
        self.classifiers = ["task"]

        # Create all entire-batch metrics
        self.batch_metrics = {dataset: {} for dataset in self.datasets}
        for domain in self.domains:
            for dataset in self.datasets:
                n = "accuracy_domain/%s/%s" % (domain, dataset)
                self.batch_metrics[dataset][n] = tf.keras.metrics.CategoricalAccuracy(name=n)
                for name in self.classifiers:
                    n = "accuracy_%s/%s/%s" % (name, domain, dataset)
                    self.batch_metrics[dataset][n] = tf.keras.metrics.CategoricalAccuracy(name=n)
        # Create all per-class metrics
        self.per_class_metrics = {dataset: {} for dataset in self.datasets}
        for i in range(self.num_classes):
            class_name = self.source_datasets[0].int_to_label(i)
            for domain in self.domains:
                for dataset in self.datasets:
                    for classifier in self.classifiers:
                        n = "accuracy_%s_class_%s/%s/%s" % (classifier, class_name, domain, dataset)
                        self.per_class_metrics[dataset][n] = tf.keras.metrics.Accuracy(name=n)
        # Variable number of losses
        self.losses = {dataset: {} for dataset in self.datasets}

    def _reset_states(self, dataset):
        """ Reset states of all the Keras metrics """
        for _, metric in self.batch_metrics[dataset].items():
            metric.reset_states()

        for _, metric in self.per_class_metrics[dataset].items():
            metric.reset_states()

        for _, metric in self.losses[dataset].items():
            metric.reset_states()

    def _process_batch(self, results, classifier, domain, dataset):
        """ Update metrics for accuracy over entire batch for domain-dataset """
        task_y_true, task_y_pred, domain_y_true, domain_y_pred = results
        # task_y_true, task_y_pred, domain_y_true, domain_y_pred, losses = results

        # Since we are now using sparse
        domain_y_true = tf.one_hot(tf.cast(domain_y_true, tf.int32), self.domain_outputs)

        domain_names = [
            "accuracy_domain/%s/%s",
        ]

        for n in domain_names:
            name = n % (domain, dataset)
            self.batch_metrics[dataset][name](domain_y_true, domain_y_pred)

        task_names = [
            "accuracy_%s/%s/%s",
            # "f1score_micro_%s/%s/%s",
            # "f1score_macro_%s/%s/%s",
            # "f1score_weighted_%s/%s/%s",
        ]

        # Since we are now using sparse
        task_y_true = tf.one_hot(tf.cast(task_y_true, tf.int32), self.num_classes)

        for n in task_names:
            name = n % (classifier, domain, dataset)
            self.batch_metrics[dataset][name](task_y_true, task_y_pred)

    def _process_per_class(self, results, classifier, domain, dataset):
        """ Update metrics for accuracy over per-class portions of batch for domain-dataset """
        # task_y_true, task_y_pred, domain_y_true, domain_y_pred, losses = results
        task_y_true, task_y_pred, domain_y_true, domain_y_pred = results
        batch_size = tf.shape(task_y_true)[0]

        # Since we are now using sparse
        task_y_true = tf.one_hot(tf.cast(task_y_true, tf.int32), self.num_classes)

        # If only predicting a single class (using softmax), then look for the
        # max value
        # e.g. [0.2 0.2 0.4 0.2] -> [0 0 1 0]
        per_class_predictions = tf.one_hot(
            tf.argmax(task_y_pred, axis=-1), self.num_classes)

        # List of per-class task metrics to update
        task_names = [
            "accuracy_%s_class_%s/%s/%s",
        ]

        for i in range(self.num_classes):
            class_name = self.source_datasets[0].int_to_label(i)

            # Get ith column (all groundtruth/predictions for ith class)
            y_true = tf.slice(task_y_true, [0, i], [batch_size, 1])  # if not sparse
            y_pred = tf.slice(per_class_predictions, [0, i], [batch_size, 1])

            # For single-class prediction, we want to first isolate which
            # examples in the batch were supposed to be class X. Then, of
            # those, calculate accuracy = correct / total.
            rows_of_class_y = tf.where(tf.equal(y_true, 1))  # i.e. have 1
            acc_y_true = tf.gather(y_true, rows_of_class_y)
            acc_y_pred = tf.gather(y_pred, rows_of_class_y)

            # Update metrics
            for n in task_names:
                name = n % (classifier, class_name, domain, dataset)
                self.per_class_metrics[dataset][name](acc_y_true, acc_y_pred)

    def _process_losses(self, results, domain, dataset):
        """ Update losses, but create if it hasn't been seen before. Create here
        rather than in __init__ since different methods have different numbers
        of losses, e.g. some have a domain loss and others don't """
        task_y_true, task_y_pred, domain_y_true, domain_y_pred, losses = results

        # If only one loss, then make it a list so we can use the same code
        # for either case
        if not isinstance(losses, list):
            losses = [losses]

        assert len(self.method.loss_names) >= len(losses), \
            "not enough loss_names defined in method"

        for i, loss in enumerate(losses):
            name = "loss/" + self.method.loss_names[1] + "/" + domain + "/" + dataset

            if name not in self.losses[dataset]:
                self.losses[dataset][name] = tf.keras.metrics.Mean(name=name)

            self.losses[dataset][name](loss)

    def _write_data(self, step, dataset, eval_time, train_time=None,
                    additional_losses=None):
        """ Write either the training or validation data """
        assert dataset in self.datasets, "unknown dataset " + str(dataset)

        # Write all the values to the file
        with self.writer.as_default():
            for key, metric in self.batch_metrics[dataset].items():
                tf.summary.scalar(key, metric.result(), step=step)

            for key, metric in self.per_class_metrics[dataset].items():
                tf.summary.scalar(key, metric.result(), step=step)

            for key, metric in self.losses[dataset].items():
                tf.summary.scalar(key, metric.result(), step=step)

            # Any other losses
            if additional_losses is not None:
                names, values = additional_losses

                for i, name in enumerate(names):
                    # If TensorFlow string (when using tf.function), get the
                    # value from it
                    if not isinstance(name, str):
                        name = name.numpy().decode("utf-8")

                    tf.summary.scalar("loss/%s" % (name), values[i], step=step)

            # Regardless of mapping/task, log times
            tf.summary.scalar("step_time/metrics/%s" % (dataset), eval_time, step=step)

            if train_time is not None:
                tf.summary.scalar("step_time/%s" % (dataset), train_time, step=step)

        # Make sure we sync to disk
        self.writer.flush()

    def train(self, data_a, data_b, step, train_time):
        """
        Evaluate the model on a batch of training data (during training,
        not at evaluation time -- use train_eval() for that)
        """
        dataset = "training"
        self._reset_states(dataset)
        t = time.time()
        self._run_batch(data_a, data_b, dataset)
        d_loss = self._run_double_batch(data_a, data_b)
        t = time.time() - t
        step = int(step)
        self._write_data(step, dataset, t, train_time, [[dataset + "_d_loss"], [d_loss[0]]])

    def train_eval(self):
        """
        Evaluate the model on the entire training dataset, for use during
        evaluation -- not at training time
        """
        dataset = "training"
        self._reset_states(dataset)

        if self.target_domain:
            target_datasets = self.method.target_train_eval_dataset
        else:
            target_datasets = None

        # At evaluation time, use eval tf.data.Dataset
        self._run_dataset(self.method.source_train_eval_datasets,
                          target_datasets, dataset)

    def tsen_train(self):
        data_a = self.method.source_train_eval_datasets
        data_aa = self.method.source_test_eval_datasets
        data_b = self.method.target_train_eval_dataset
        data_bb = self.method.target_test_eval_dataset
        dataset = "training"
        target_fe_all = None
        target_y_all = None
        if self.target_domain and data_b is not None:
            target_iter = iter(data_b)
            while True:
                try:
                    data = self.method.get_next_batch_single(
                        next(target_iter), is_target=True)
                    fe_out, y_true = self.method.tsne_step(data, True)
                    if target_fe_all is not None:
                        target_fe_all = tf.concat([target_fe_all, fe_out], axis=0)
                        target_y_all = tf.concat([target_y_all, y_true], axis=0)
                    else:
                        target_fe_all = fe_out
                        target_y_all = y_true
                except StopIteration:
                    break
        if self.target_domain and data_bb is not None:
            target_iter = iter(data_bb)
            while True:
                try:
                    data = self.method.get_next_batch_single(
                        next(target_iter), is_target=True)
                    fe_out, y_true = self.method.tsne_step(data, True)
                    if target_fe_all is not None:
                        target_fe_all = tf.concat([target_fe_all, fe_out], axis=0)
                        target_y_all = tf.concat([target_y_all, y_true], axis=0)
                    else:
                        target_fe_all = fe_out
                        target_y_all = y_true
                except StopIteration:
                    break

        source_fe_all = None
        source_y_all = None
        if data_a is not None:
            source_iterators = [iter(x) for x in data_a]
            for i, source_iter in enumerate(source_iterators):
                while True:
                    try:
                        data = self.method.get_next_batch_single(
                            next(source_iter), is_target=False, index=i)
                        fe_out, y_true = self.method.tsne_step(data, False)
                        if source_fe_all is not None:
                            source_fe_all = tf.concat([source_fe_all, fe_out], axis=0)
                            source_y_all = tf.concat([source_y_all, y_true], axis=0)
                        else:
                            source_fe_all = fe_out
                            source_y_all = y_true
                        print()
                    except StopIteration:
                        break
        if data_aa is not None:
            source_iterators = [iter(x) for x in data_aa]
            for i, source_iter in enumerate(source_iterators):
                while True:
                    try:
                        data = self.method.get_next_batch_single(
                            next(source_iter), is_target=False, index=i)
                        fe_out, y_true = self.method.tsne_step(data, False)
                        if source_fe_all is not None:
                            source_fe_all = tf.concat([source_fe_all, fe_out], axis=0)
                            source_y_all = tf.concat([source_y_all, y_true], axis=0)
                        else:
                            source_fe_all = fe_out
                            source_y_all = y_true
                        print()
                    except StopIteration:
                        break
        return source_fe_all.numpy(), source_y_all.numpy(), target_fe_all.numpy(), target_y_all.numpy()

    def test(self, step=None, evaluation=False):
        """
        Evaluate the model on domain A/B but batched to make sure we don't run
        out of memory

        Note: leave off step if evaluation=True

        Returns: source task validation accuracy, target task validation accuracy
        """
        dataset = "validation"
        self._reset_states(dataset)
        t = time.time()

        if self.target_domain:
            target_datasets = self.method.target_test_eval_dataset
        else:
            target_datasets = None

        self._run_dataset(self.method.source_test_eval_datasets,
                          target_datasets, dataset)

        # We use the validation accuracy to save the best model
        acc_source = self.batch_metrics["validation"]["accuracy_task/source/validation"]
        validation_accuracy_source = float(acc_source.result())

        if self.target_domain:
            acc_target = self.batch_metrics["validation"]["accuracy_task/target/validation"]
            validation_accuracy_target = float(acc_target.result())
        else:
            validation_accuracy_target = None

        t = time.time() - t

        if not evaluation:
            assert step is not None, "Must pass step to test() if evaluation=False"
            step = int(step)
            self._write_data(step, dataset, t)

        return validation_accuracy_source, validation_accuracy_target

    def _run_batch(self, data_a, data_b, dataset):
        """ Run a single batch of A/B data through the model -- data_a and data_b
        should both be a tuple of (x, task_y_true, domain_y_true) """
        if data_a is not None:
            self._run_single_batch(data_a, dataset, "source")

        if self.target_domain and data_b is not None:
            self._run_single_batch(data_b, dataset, "target")

    def _run_single_batch(self, data, dataset_name, domain_name):
        """ Run a single batch of data through the model """
        assert dataset_name in self.datasets, "unknown dataset " + str(dataset_name)
        assert domain_name in self.domains, "unknown domain " + str(domain_name)

        is_target = domain_name == "target"
        results = self.method.eval_step(data, is_target)

        classifier = "task"  # Which classifier's task_y_pred are we looking at?
        self._process_batch(results, classifier, domain_name, dataset_name)
        self._process_per_class(results, classifier, domain_name, dataset_name)
        # self._process_losses(results, domain_name, dataset_name)

    def _run_double_batch(self, data_a, data_b):
        x_a, y_a, domain_a = data_a
        x_b, y_b, domain_b = data_b

        if not isinstance(x_a, list):
            x_a = [x_a]
        if not isinstance(y_a, list):
            y_a = [y_a]
        if not isinstance(domain_a, list):
            domain_a = [domain_a]

        if not isinstance(x_b, list):
            x_b = [x_b]
        if not isinstance(y_b, list):
            y_b = [y_b]
        if not isinstance(domain_b, list):
            domain_b = [domain_b]

        x_a, orig_task_y_true_a, orig_domain_y_true_a = self.method.prepare_data_eval((x_a, y_a, domain_a))
        x_b, orig_task_y_true_b, orig_domain_y_true_b = self.method.prepare_data_eval((x_b, y_b, domain_b))
        x = tf.concat((x_a, x_b), axis=0)
        task_y_true = tf.concat((orig_task_y_true_a, orig_task_y_true_b), axis=0)
        domain_y_true = tf.concat((orig_domain_y_true_a, orig_domain_y_true_b), axis=0)
        losses_list = []
        for i in range(self.method.ensemble_size):
            # Run through model
            task_y_pred, domain_y_pred, fe_output = self.method.call_model(x,
                                                                           which_model=i, training=False)

            # Calculate losses
            losses = self.method.compute_losses(x, task_y_true,
                                                domain_y_true, task_y_pred, domain_y_pred, fe_output,
                                                which_model=i, training=False)

            if not isinstance(losses, list):
                losses = [losses]
            else:
                losses = [losses[2]]

            losses_list.append(losses)

        losses_avg = self.method.add_multiple_losses(losses_list, average=True)

        return losses_avg

    def _run_dataset(self, data_a, data_b, dataset):
        """ Run all the data A/B through the model -- data_a and data_b
        should both be of type tf.data.Dataset (with data_a a list of them) """
        if data_a is not None:
            source_iterators = [iter(x) for x in data_a]

            # Do each source domain individually since they may have different
            # amounts of data and this loop will break as soon as the smallest
            # one has no data.
            for i, source_iter in enumerate(source_iterators):
                while True:
                    try:
                        data = self.method.get_next_batch_single(
                            next(source_iter), is_target=False, index=i)
                        self._run_single_batch(data, dataset, "source")
                    except StopIteration:
                        break

        if self.target_domain and data_b is not None:
            target_iter = iter(data_b)

            while True:
                try:
                    data = self.method.get_next_batch_single(
                        next(target_iter), is_target=True)
                    self._run_single_batch(data, dataset, "target")
                except StopIteration:
                    break

    def results(self):
        """ Returns one dictionary of all the current metric results (floats) """
        results = {}

        for dataset in self.datasets:
            for key, metric in self.batch_metrics[dataset].items():
                results[key] = float(metric.result())

            for key, metric in self.per_class_metrics[dataset].items():
                results[key] = float(metric.result())

            for key, metric in self.losses[dataset].items():
                results[key] = float(metric.result())

        return results
