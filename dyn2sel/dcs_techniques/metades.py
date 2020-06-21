from dyn2sel.dcs_techniques.deslib_interface import DESLIBInterface
import deslib.des as deslib


class METADES(DESLIBInterface):
    def set_stencil(self):
        return deslib.METADES
