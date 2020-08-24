from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.dcs as deslib


class APriori(DESLIBInterface):
    """
    A Priori
    The A Priori method provides a selection method based on probabilities. It gathers the most similar instances in
    the validation set to the instance to be classified using K-Nearest Neighbors. Then, for each classifier, it sums
    the probability outputted by the calssifier for each neighbor being of the class they really are. The classifier
    with the highest value for this metric is selected for prediction.

    Parameters
    ----------
    **kwargs
        Defined `here <https://deslib.readthedocs.io/en/latest/modules/dcs/a_priori.html>`

    The implementation used for this method is provided by `DESLIB <https://deslib.readthedocs.io/>`_

    References
    ----------
        G. Giacinto and F. Roli, Methods for Dynamic Classifier Selection 10th Int. Conf. on Image Anal. and Proc.,
        Venice, Italy (1999), 659-664.
    """

    def _get_stencil(self):
        return deslib.APriori
