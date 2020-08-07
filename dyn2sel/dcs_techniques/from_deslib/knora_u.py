from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.des as deslib


class KNORAU(DESLIBInterface):
    """
    KNORA-UNION

    K-Nearest Oracles Union or KNORA-UNION looks for classifiers in the ensemble that classified at least one of the
    K-nearest neighbors correctly.  If there is no classifier with such level of accuracy in the neighbors,
    the classifiers that got (K−1) instances corrected are selected, and subsequently decreasing K until at least one
    of the classifiers is selected. The classifiers that predicted more neighbors correctly have more votes on the
    final combination rule

    Parameters
    ----------
    **kwargs
        Defined `here <https://deslib.readthedocs.io/en/latest/modules/des/knora_u.html>`

    The implementation used for this method is provided by `DESLIB <https://deslib.readthedocs.io/>`_

    References
    ----------
        Ko, Albert HR, Robert Sabourin, and Alceu Souza Britto Jr. “From dynamic classifier selection to dynamic
        ensemble selection.” Pattern Recognition 41.5 (2008): 1718-1731.

    """

    def _get_stencil(self):
        return deslib.KNORAU
