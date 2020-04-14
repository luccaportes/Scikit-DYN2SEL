from dyn2sel.apply_dcs import DYNSEMethod, WPSMethod, WPSMethodPostDrift
from dyn2sel.dcs_techniques import KNORAE
from skmultiflow.meta import AdaptiveRandomForest, OzaBagging
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import SEAGenerator, ConceptDriftStream, AGRAWALGenerator
from skmultiflow.evaluation import EvaluatePrequential

sea = ConceptDriftStream(
    AGRAWALGenerator(classification_function=0, random_state=42),
    AGRAWALGenerator(classification_function=3, random_state=42),
    position=2000, width=1, random_state=42)
sea.prepare_for_use()

# w = WPSMethod(AdaptiveRandomForest(n_estimators=10), 5, window_size=300, metric="recall")
# w.partial_fit(X_train, y_train)
# w.predict(X_test)

w = WPSMethodPostDrift(AdaptiveRandomForest(n_estimators=3, random_state=42), 2, window_size=100, metric="recall")

# d = DYNSEMethod(AdaptiveRandomForest(), 1000, KNORAE())
# d = DYNSEMethod(HoeffdingTree(), 1000, KNORAE())
rf = AdaptiveRandomForest(n_estimators=3, random_state=42)
# #
ev = EvaluatePrequential(
    n_wait=1000, max_samples=5000, pretrain_size=0,
    metrics=["accuracy", "model_size"], output_file="teste.txt")
ev.evaluate(sea, [w, rf], [ "WPS", "RF"])
# ev.evaluate(sea, [rf], [ "ARF"])



# X_train, y_train = sea.next_sample(50)
# X_test, y_test = sea.next_sample(1)

# w.partial_fit(X_train, y_train)
# print(w.predict(X_test))

# X_train, y_train = sea.next_sample(2000)
# X_test, y_test = sea.next_sample(200)
# rf.partial_fit(X_train, y_train)
# print(rf.predict(X_test))

