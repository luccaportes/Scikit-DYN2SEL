import sys
sys.path.append("..")
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential as mtflowPrequential
from dyn2sel.utils.evaluators import EvaluatePrequential as dyn2selPrequential
from skmultiflow.bayes import NaiveBayes


def test_equality_multiflow():
    gen = SEAGenerator(random_state=42)
    gen.prepare_for_use()
    evaluator_mtflow = mtflowPrequential(max_samples=1000, pretrain_size=0, restart_stream=True)
    evaluator_dyn2sel = dyn2selPrequential(max_samples=1000, pretrain_size=0)
    nb_mtflow = evaluator_mtflow.evaluate(gen, NaiveBayes())[0].__dict__
    nb_dyn2sel = evaluator_dyn2sel.evaluate(gen, NaiveBayes())[0].__dict__
    del nb_mtflow['_attribute_observers']
    del nb_dyn2sel['_attribute_observers']
    assert nb_mtflow == nb_dyn2sel
