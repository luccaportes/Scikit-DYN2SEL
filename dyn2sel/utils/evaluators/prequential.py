import skmultiflow.evaluation as sk_ev
from timeit import default_timer as timer
from numpy import unique
from skmultiflow.utils import constants


class EvaluatePrequential(sk_ev.EvaluatePrequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_oracle = None

    def _train_and_test(self):
        """Method to control the prequential evaluation.
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.
        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.
        """
        self._start_time = timer()
        self._end_time = timer()
        print("Prequential Evaluation")
        print("Evaluating {} target(s).".format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True
        if self.pretrain_size > 0:
            print("Pre-training on {} sample(s).".format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(
                        X=X, y=y, classes=self.stream.target_values
                    )
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(
                        X=X, y=y, classes=unique(self.stream.target_values)
                    )
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(
                    self.pretrain_size
                )
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        print("Evaluating...")
        while (
            (self.global_sample_count < actual_max_samples)
            & (self._end_time - self._start_time < self.max_time)
            & (self.stream.has_more_samples())
        ):
            try:
                X, y = self.stream.next_sample(self.batch_size)

                if X is not None and y is not None:
                    if self.is_oracle is None:
                        self._init_is_oracle()
                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        try:
                            # Testing time
                            self.running_time_measurements[
                                i
                            ].compute_testing_time_begin()
                            if self.is_oracle[i]:
                                prediction[i].extend(self.model[i].predict(X, y))
                            else:
                                prediction[i].extend(self.model[i].predict(X))

                            self.running_time_measurements[i].compute_testing_time_end()
                        except TypeError:
                            raise TypeError(
                                "Unexpected prediction value from {}".format(
                                    type(self.model[i]).__name__
                                )
                            )
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(
                                y[i], prediction[j][i]
                            )
                            self.current_eval_measurements[j].add_result(
                                y[i], prediction[j][i]
                            )
                    self._check_progress(actual_max_samples)

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if (
                                self._task_type != constants.REGRESSION
                                and self._task_type != constants.MULTI_TARGET_REGRESSION
                            ):
                                # Accounts for the moment of training beginning
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_begin()
                                self.model[i].partial_fit(
                                    X, y, self.stream.target_values
                                )
                                # Accounts the ending of training
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_end()
                            else:
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(
                                self.batch_size
                            )
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_time_measurements[
                                i
                            ].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[
                                i
                            ].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(
                                self.batch_size
                            )

                    if (
                        (self.global_sample_count % self.n_wait) == 0
                        or (self.global_sample_count >= actual_max_samples)
                        or (self.global_sample_count / self.n_wait > update_count + 1)
                    ):
                        if prediction is not None:
                            self._update_metrics()
                        update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print("Done")

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def _init_is_oracle(self):
        self.is_oracle = []
        for i in range(self.n_models):
            try:
                self.model[i]._is_oracle()
                self.is_oracle.append(True)
            except AttributeError:
                self.is_oracle.append(False)
