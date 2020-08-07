from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.dcs as deslib


class MCB(DESLIBInterface):
    """
    LCA
    The Local Class Accuracy (LCA) first gathers the K-Nearest neighbors of the query instance that are from the class
    predicted by each classifier. Next, the algorithm computes the accuracy of each classifier regarding only the
    neighbors returned. It then proceeds to pick the most accurate classifier to predict the instance.

    Parameters
    ----------
    **kwargs
        Defined `here <https://deslib.readthedocs.io/en/latest/modules/dcs/a_posteriori.html>`

    The implementation used for this method is provided by `DESLIB <https://deslib.readthedocs.io/>`_

    References
    ----------
        Woods, Kevin, W. Philip Kegelmeyer, and Kevin Bowyer. “Combination of multiple classifiers using local accuracy
        estimates.” IEEE transactions on pattern analysis and machine intelligence 19.4 (1997): 405-410.
    """

    def _get_stencil(self):
        return deslib.MCB
