from dyn2sel.validation_set import ValidationSet
from skmultiflow.data import SEAGenerator


def test_size():
    val_set = ValidationSet(max_size=100)
    gen = SEAGenerator()
    gen.prepare_for_use()
    X, y = gen.next_sample(20)
    val_set.add_instances(X, y)
    assert len(val_set.buffer_X) == 20
    assert len(val_set.buffer_y) == 20
    X, y = gen.next_sample(80)
    val_set.add_instances(X, y)
    assert len(val_set.buffer_X) == 100
    assert len(val_set.buffer_y) == 100
    X, y = gen.next_sample(10)
    val_set.add_instances(X, y)
    assert len(val_set.buffer_X) == 100
    assert len(val_set.buffer_y) == 100
    val_set.clear()
    assert len(val_set.buffer_X) == 0
    assert len(val_set.buffer_y) == 0


