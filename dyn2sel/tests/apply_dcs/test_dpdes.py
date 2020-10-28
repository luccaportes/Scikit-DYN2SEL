from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes
from dyn2sel.apply_dcs import DPDESMethod
from dyn2sel.dcs_techniques import METADES, ModifiedRank


def test_ensemble_size():
    # since each member of the ensemble is initialized when the number of instances reach the chunk size, the size of
    # the ensemble should n_samples // chunk_size
    chunk_size = 100
    n_samples = 1050
    gen = SEAGenerator()
    # gen.prepare_for_use()
    dynse = DPDESMethod(NaiveBayes(), chunk_size, ModifiedRank())
    X, y = gen.next_sample(n_samples)
    dynse.partial_fit(X, y)
    assert len(dynse.ensemble) == n_samples // chunk_size


def test_accuracy():
    # an ensemble of Naive Bayes should perform at the very least 85% with 200 instances of SEAGenerator
    chunk_size = 100
    n_samples_train = 1050
    n_samples_test = 200
    gen = SEAGenerator(noise_percentage=0.0)
    # gen.prepare_for_use()
    nb = NaiveBayes()
    mde = DPDESMethod(nb, chunk_size, METADES())
    X_train, y_train = gen.next_sample(n_samples_train)
    X_test, y_test = gen.next_sample(n_samples_test)
    mde.partial_fit(X_train, y_train)
    assert mde.score(X_test, y_test) > 0.85
