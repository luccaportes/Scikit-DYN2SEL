from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.des as deslib


class METADES(DESLIBInterface):
    """
    METADES
    The META-DES technique is a different approach to the classifier selection problem. The previous method focused on
    single features of the relationship between each classifier and the test instances, for example, Local Accuracy.
    META-DES, instead, takes into consideration multiple features. And instead of trying to design a selection method,
    it considers this task another classification problem, thus delegating it to a meta-classifier.

    Parameters
    ----------
    **kwargs
        Defined `here <https://deslib.readthedocs.io/en/latest/modules/dcs/a_posteriori.html>`

    The implementation used for this method is provided by `DESLIB <https://deslib.readthedocs.io/>`_

    References
    ----------
        Cruz, R.M., Sabourin, R., Cavalcanti, G.D. and Ren, T.I., 2015. META-DES: A dynamic ensemble selection
        framework using meta-learning. Pattern Recognition, 48(5), pp.1925-1935.
    """

    def _get_stencil(self):
        return deslib.METADES
