from abc import abstractmethod

from skmultiflow.core import ClassifierMixin


class DCSApplier(ClassifierMixin):
    """
    DCSApplier base class for compatibility with scikit-multiflow.
    """

    @abstractmethod
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Partially (incrementally) fit the model.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.
        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. Usage varies depending
            on the learning method.
        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.
            Usage varies depending on the learning method.
        Returns
        -------
        self
        Notes
        -------
        Description taken from scikit-multiflow
        """
        pass

    @abstractmethod
    def predict(self, X, y=None):
        """
        Predict classes for the passed data.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the class labels for.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X. This parameter is only considered with the oracle selector.
        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.
        Notes
        -------
        - Description partially taken from scikit-multiflow
        - This method signature is overwritten because we need to pass the y parameter for the predict method when using
         the oracle selector
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        Notes
        -------
        - Description partially taken from scikit-multiflow
        - This method is overwritten because we need to pass the y parameter for the predict method for the oracle
        selector to work
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X, y), sample_weight=sample_weight)

    def _is_oracle(self):
        return self.dcs_method._is_oracle()
