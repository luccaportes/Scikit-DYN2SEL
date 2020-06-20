from dyn2sel.apply_dcs import DYNSEMethod
from dyn2sel.dcs_techniques import KNORAE
from skmultiflow.meta import OzaBagging
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential

clf = DYNSEMethod(NaiveBayes(), chunk_size=100, dcs_method=KNORAE())
gen = SEAGenerator(random_state=4482)
X, y = gen.next_sample(302)
clf.partial_fit(X, y)
X, y = gen.next_sample(15)
clf.predict(X)
#
# ev = EvaluatePrequential(n_wait=200, max_samples=2000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)

# clf = OzaBagging(NaiveBayes(), n_estimators=20)
#
# ev = EvaluatePrequential(n_wait=200, max_samples=2000, pretrain_size=0)
#
# gen = SEAGenerator(random_state=52)
# ev.evaluate(gen, clf)