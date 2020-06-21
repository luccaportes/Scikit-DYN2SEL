from dyn2sel.apply_dcs import DYNSEMethod, MDEMethod
from dyn2sel.dcs_techniques import KNORAE, KNORAU, OLA, LCA, APriori,\
    Rank, ModifiedRank, KNOP, METADES, MCB, APosteriori
from skmultiflow.meta import OzaBagging, AdaptiveRandomForest
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

clf = MDEMethod(NaiveBayes(), chunk_size=100)
gen = SEAGenerator(random_state=4482)
X, y = gen.next_sample(302)
clf.partial_fit(X, y)
X, y = gen.next_sample(600)
clf.predict(X)
print(clf.score(X, y))
#
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