from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.des as deslib


class KNOP(DESLIBInterface):
    """
    KNOP
    K-Nearest Output Profiles (KNOP, gathers the most similar instances in the validation set using K-Nearest Neighbors
    using their output profiles on the instance to be predicted. The next step follows KNORA-U, i.e, if the classifier
    correctly predicts at least one of the neighbors, it is then picked to classify the instance and the more correctly
    predicted neighbors, the more votes the classifier will have in the final prediction.

    Parameters
    ----------
    **kwargs
        Defined `here <https://deslib.readthedocs.io/en/latest/modules/dcs/a_posteriori.html>`

    The implementation used for this method is provided by `DESLIB <https://deslib.readthedocs.io/>`_

    References
    ----------
        Cavalin, Paulo R., Robert Sabourin, and Ching Y. Suen. “LoGID: An adaptive framework combining local and global
        incremental learning for dynamic selection of ensembles of HMMs.” Pattern Recognition 45.9 (2012): 3544-3556.
    """

    def _get_stencil(self):
        return deslib.KNOP
