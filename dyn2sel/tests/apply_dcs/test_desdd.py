import sys

sys.path.append("..")
from skmultiflow.data import SEAGenerator

# from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import AdaptiveRandomForest, OzaBagging
from dyn2sel.apply_dcs import DESDDMethod


def test_accuracy():
    # an ensemble of Adaptive Random Forests should perform at the very least 80% with 200 instances of SEAGenerator
    n_samples_train = 200
    n_samples_test = 200
    gen = SEAGenerator(noise_percentage=0.0)
    # gen.prepare_for_use()
    arf = AdaptiveRandomForest()
    desdd = DESDDMethod(arf)
    X_train, y_train = gen.next_sample(n_samples_train)
    X_test, y_test = gen.next_sample(n_samples_test)
    desdd.partial_fit(X_train, y_train)
    assert desdd.score(X_test, y_test) > 0.80


# def test_drift():
#     oza = OzaBagging(NaiveBayes())
#     desdd = DESDDMethod(oza, max_lambda=10)
#     gen = SEAGenerator(classification_function=0)
#     gen.prepare_for_use()
#     X_pre_drift, y_pre_drift = gen.next_sample(200)
#     gen = SEAGenerator(classification_function=3)
#     gen.prepare_for_use()
#     X_post_drift, y_post_drift = gen.next_sample(200)
#
#     desdd.partial_fit(X_pre_drift, y_pre_drift, classes=[0, 1])
#     old_lambdas = desdd.ensemble.lambdas
#
#     desdd.partial_fit(X_post_drift, y_post_drift)
#     new_lambdas = desdd.ensemble.lambdas
#     o=9
