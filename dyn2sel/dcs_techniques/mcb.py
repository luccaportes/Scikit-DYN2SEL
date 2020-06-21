from dyn2sel.dcs_techniques.deslib_interface import DESLIBInterface
import deslib.dcs as deslib


class MCB(DESLIBInterface):
    def set_stencil(self):
        return deslib.MCB