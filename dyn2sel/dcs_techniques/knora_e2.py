from dyn2sel.dcs_techniques.deslib_interface import DESLIBInterface
import deslib.des as deslib


class KNORAE2(DESLIBInterface):
    def set_stencil(self):
        return deslib.KNORAE