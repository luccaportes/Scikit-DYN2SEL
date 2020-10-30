from dyn2sel.apply_dcs import DYNSEMethod
from dyn2sel.dcs_techniques import KNORAU
from skmultiflow.data import RandomTreeGenerator, ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import OzaBagging

generator = ConceptDriftStream(
    stream=RandomTreeGenerator(sample_random_state=42, tree_random_state=42),
    drift_stream=RandomTreeGenerator(sample_random_state=43, tree_random_state=43),
    position=2500,
    width=1,
)
dynse = DYNSEMethod(NaiveBayes(), 200, KNORAU(), max_ensemble_size=10)
ozabag = OzaBagging(NaiveBayes(), n_estimators=10)

evaluator = EvaluatePrequential(
    max_samples=5000, n_wait=200, batch_size=200, pretrain_size=0
)
evaluator.evaluate(generator, [dynse, ozabag], ["Dynse", "Ozabag"])
