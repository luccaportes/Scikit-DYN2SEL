import sys
sys.path.append("..")
from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes
from dyn2sel.apply_dcs import MDEMethod
from dyn2sel.dcs_techniques import KNORAE

def test_ensemble_size():
    # since each member of the ensemble is initialized when the number of instances reach the chunk size, the size of
    # the ensemble should n_samples // chunk_size
    chunk_size = 100
    n_samples = 1050
    gen = SEAGenerator()
    gen.prepare_for_use()
    dynse = MDEMethod(NaiveBayes(), chunk_size, KNORAE())
    X, y = gen.next_sample(n_samples)
    dynse.partial_fit(X, y)
    assert len(dynse.ensemble) == n_samples // chunk_size