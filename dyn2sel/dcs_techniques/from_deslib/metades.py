from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.des as deslib


class METADES(DESLIBInterface):
    def _get_stencil(self):
        return deslib.METADES
