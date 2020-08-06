import sys
sys.path.append("..")
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForest
from dyn2sel.apply_dcs import DESDDMethod


def test_accuracy():
    # an ensemble of Adaptive Random Forests should perform at the very least 80% with 200 instances of SEAGenerator
    n_samples_train = 200
    n_samples_test = 200
    gen = SEAGenerator()
    gen.prepare_for_use()
    arf = AdaptiveRandomForest()
    desdd = DESDDMethod(arf)
    X_train, y_train = gen.next_sample(n_samples_train)
    X_test, y_test = gen.next_sample(n_samples_test)
    desdd.partial_fit(X_train, y_train)
    assert desdd.score(X_test, y_test) > 0.80
