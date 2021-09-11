import skmultiflow.evaluation as sk_ev
from timeit import default_timer as timer
from numpy import unique
from skmultiflow.utils import constants, get_dimensions


class EvaluateHoldout(sk_ev.EvaluateHoldout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_oracle = None

    def _periodic_holdout(self):
        """Method to control the holdout evaluation."""
        self._start_time = timer()
        self._end_time = timer()
        print("Holdout Evaluation")
        print("Evaluating {} target(s).".format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True

        if not self.dynamic_test_set:
            print("Separating {} holdout samples.".format(self.test_size))
            self.X_test, self.y_test = self.stream.next_sample(self.test_size)
            self.global_sample_count += self.test_size

        performance_sampling_cnt = 0
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
                    self.global_sample_count += self.batch_size

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type == constants.CLASSIFICATION:
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_begin()
                                self.model[i].partial_fit(
                                    X=X, y=y, classes=self.stream.target_values
                                )
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_end()
                            elif (
                                self._task_type == constants.MULTI_TARGET_CLASSIFICATION
                            ):
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_begin()
                                self.model[i].partial_fit(
                                    X=X, y=y, classes=unique(self.stream.target_values)
                                )
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_end()
                            else:
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_begin()
                                self.model[i].partial_fit(X=X, y=y)
                                self.running_time_measurements[
                                    i
                                ].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(
                                self.batch_size
                            )
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            # Compute running time
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

                    self._check_progress(actual_max_samples)  # TODO Confirm place

                    # Test on holdout set
                    if self.dynamic_test_set:
                        perform_test = self.global_sample_count >= (
                            self.n_wait * (performance_sampling_cnt + 1)
                            + (self.test_size * performance_sampling_cnt)
                        )
                    else:
                        perform_test = (
                            self.global_sample_count - self.test_size
                        ) % self.n_wait == 0

                    if perform_test | (self.global_sample_count >= actual_max_samples):

                        if self.dynamic_test_set:
                            print(
                                "Separating {} holdout samples.".format(self.test_size)
                            )
                            self.X_test, self.y_test = self.stream.next_sample(
                                self.test_size
                            )
                            self.global_sample_count += get_dimensions(self.X_test)[0]

                        # Test
                        if (self.X_test is not None) and (self.y_test is not None):
                            prediction = [[] for _ in range(self.n_models)]
                            for i in range(self.n_models):
                                try:
                                    self.running_time_measurements[
                                        i
                                    ].compute_testing_time_begin()
                                    if self.is_oracle[i]:
                                        prediction[i].extend(
                                            self.model[i].predict(
                                                self.X_test, self.y_test
                                            )
                                        )
                                    else:
                                        prediction[i].extend(
                                            self.model[i].predict(self.X_test)
                                        )
                                    self.running_time_measurements[
                                        i
                                    ].compute_testing_time_end()
                                    self.running_time_measurements[
                                        i
                                    ].update_time_measurements(self.test_size)
                                except TypeError:
                                    raise TypeError(
                                        "Unexpected prediction value from {}".format(
                                            type(self.model[i]).__name__
                                        )
                                    )
                            if prediction is not None:
                                for j in range(self.n_models):
                                    for i in range(len(prediction[0])):
                                        self.mean_eval_measurements[j].add_result(
                                            self.y_test[i], prediction[j][i]
                                        )
                                        self.current_eval_measurements[j].add_result(
                                            self.y_test[i], prediction[j][i]
                                        )

                                self._update_metrics()
                            performance_sampling_cnt += 1

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        self.evaluation_summary()

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
