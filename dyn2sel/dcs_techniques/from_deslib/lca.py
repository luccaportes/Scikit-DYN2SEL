from dyn2sel.dcs_techniques.from_deslib.deslib_interface import DESLIBInterface
import deslib.dcs as deslib


class LCA(DESLIBInterface):
    def _get_stencil(self):
        return deslib.LCA
