from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.dcs as deslib


class ModifiedRank(DESLIBInterface):
    """
    Modified Rank
    Proposed modified version of DCS-RANK. When an instance arrives to be predicted, for each classifier, the nearest
    instances that belong to the class that was predicted by the classifier are returned. Finally, the neighbors of
    that class are sorted according to the Euclidean distance to the instance to be predicted. Next, the rank of
    competence of the classifiers is built, based on the number of consecutive correct predictions each classifier
    made on the list of neighbors.

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
        return deslib.Rank
