from dyn2sel.apply_dcs import DYNSEMethod, MDEMethod, DESDDMethod
from dyn2sel.dcs_techniques import KNORAE, KNORAU, OLA, LCA, APriori,\
    Rank, ModifiedRank, KNOP, METADES, MCB, APosteriori, KNORAE2
from skmultiflow.meta import OzaBagging, AdaptiveRandomForest
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.metrics import ClassificationPerformanceEvaluator
import timeit

setup_1 = """
from dyn2sel.dcs_techniques import KNORAE, KNORAE2
from skmultiflow.bayes import NaiveBayes
from dyn2sel.apply_dcs import DYNSEMethod
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import SEAGenerator
clf = DYNSEMethod(NaiveBayes(), chunk_size=100, dcs_method=KNORAE2(), max_ensemble_size=10)
ev = EvaluatePrequential(n_wait=100, max_samples=500, pretrain_size=0)
gen = SEAGenerator(random_state=52)
"""

setup_2 = """
from dyn2sel.dcs_techniques import KNORAE, KNORAE2
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential
from dyn2sel.apply_dcs import DYNSEMethod
from skmultiflow.data import SEAGenerator
clf = DYNSEMethod(NaiveBayes(), chunk_size=100, dcs_method=KNORAE(), max_ensemble_size=10)
ev = EvaluatePrequential(n_wait=100, max_samples=500, pretrain_size=0)
gen = SEAGenerator(random_state=52)
"""

run = "X, y = gen.next_sample(500);clf.partial_fit(X,y);X, y = gen.next_sample(1000);clf.predict(X)"

a = timeit.timeit(run, setup_1, number=5)
b = timeit.timeit(run, setup_2, number=5)

print("deslib", a)
print("mine", b)






# clf = DYNSEMethod(HoeffdingTree(), chunk_size=1000, dcs_method=KNORAE(), max_ensemble_size=10)
# ev = EvaluatePrequential(n_wait=1000, max_samples=10000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)
#


# arf = OzaBagging()
# gen = SEAGenerator(random_state=4482)
# X, y = gen.next_sample(302)
# arf.partial_fit(X, y, classes=[0,1])
# print(arf.ensemble[0].get_info())
# print()

# clf = DESDDMethod(AdaptiveRandomForest())
# gen = SEAGenerator(random_state=4482)
# X, y = gen.next_sample(10)
# clf.partial_fit(X, y)
# X, y = gen.next_sample(12)
# clf.partial_fit(X, y)
# X, y = gen.next_sample(600)
# clf.predict(X)
# print(clf.score(X, y))
# #
# clf = DYNSEMethod(HoeffdingTree(), chunk_size=1000, dcs_method=A())
# ev = EvaluatePrequential(n_wait=1000, max_samples=10000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)
#
#
# clf = DYNSEMethod(HoeffdingTree(), chunk_size=1000, dcs_method=KNORAE())
# ev = EvaluatePrequential(n_wait=1000, max_samples=10000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)
#
#
#
# clf = DYNSEMethod(HoeffdingTree(), chunk_size=1000, dcs_method=OLA())
# ev = EvaluatePrequential(n_wait=1000, max_samples=10000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)
#
#
# clf = DYNSEMethod(HoeffdingTree(), chunk_size=1000, dcs_method=LCA())
# ev = EvaluatePrequential(n_wait=1000, max_samples=10000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)
#
# clf = OzaBagging(HoeffdingTree(), n_estimators=10)
#
# ev = EvaluatePrequential(n_wait=1000, max_samples=10000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)
#
# clf = AdaptiveRandomForest(n_estimators=10)
#
# ev = EvaluatePrequential(n_wait=1000, max_samples=10000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)