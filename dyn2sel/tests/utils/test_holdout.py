from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluateHoldout as mtflowHoldout
from dyn2sel.utils.evaluators import EvaluateHoldout as dyn2selHoldout
from skmultiflow.bayes import NaiveBayes


def test_equality_multiflow():
    gen = SEAGenerator(random_state=42)
    gen.prepare_for_use()
    evaluator_mtflow = mtflowHoldout(n_wait=100, max_samples=1000, test_size=100, restart_stream=True)
    evaluator_dyn2sel = dyn2selHoldout(n_wait=100, max_samples=1000, test_size=100)
    nb_mtflow = evaluator_mtflow.evaluate(gen, NaiveBayes())[0].__dict__
    nb_dyn2sel = evaluator_dyn2sel.evaluate(gen, NaiveBayes())[0].__dict__
    del nb_mtflow['_attribute_observers']
    del nb_dyn2sel['_attribute_observers']
    assert nb_mtflow == nb_dyn2sel
