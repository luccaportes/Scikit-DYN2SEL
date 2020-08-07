from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluateHoldout as mtflowHoldout
from dyn2sel.utils.evaluators import EvaluateHoldout as dyn2selHoldout
from dyn2sel.dcs_techniques import Oracle, Rank
from skmultiflow.bayes import NaiveBayes
from io import StringIO
from contextlib import redirect_stdout
from dyn2sel.apply_dcs import DYNSEMethod



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

def test_oracle_better():
    gen = SEAGenerator(random_state=42)
    gen.prepare_for_use()
    evaluator_dyn2sel = dyn2selHoldout(n_wait=100, max_samples=1000, test_size=100)
    dynse_rank = DYNSEMethod(NaiveBayes(), 100, Rank())
    f = StringIO()
    with redirect_stdout(f):
        evaluator_dyn2sel.evaluate(gen, dynse_rank)
    out = f.getvalue()
    f.close()
    acc_rank = out[out.find("Accuracy"):]
    acc_rank = acc_rank[acc_rank.find(":")+2:]
    acc_rank = acc_rank[:acc_rank.find("\n")]
    acc_rank = float(acc_rank)

    evaluator_dyn2sel = dyn2selHoldout(n_wait=100, max_samples=1000, test_size=100)
    dynse_oracle = DYNSEMethod(NaiveBayes(), 100, Oracle())
    f = StringIO()
    with redirect_stdout(f):
        evaluator_dyn2sel.evaluate(gen, dynse_oracle)
    out = f.getvalue()
    f.close()
    acc_oracle = out[out.find("Accuracy"):]
    acc_oracle = acc_oracle[acc_oracle.find(":") + 2:]
    acc_oracle = acc_oracle[:acc_oracle.find("\n")]
    acc_oracle = float(acc_oracle)

    assert acc_oracle > acc_rank