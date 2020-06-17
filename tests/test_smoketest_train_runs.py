from types import SimpleNamespace

import pytest
import yaml

import amci.train


class DummyClass:
    def __getattribute__(self, name):
        return lambda *args, **kwargs: None


@pytest.mark.parametrize("q1_or_q2", ("q1", "q2"))
@pytest.mark.parametrize("tail_integral_d", (1, 5))
def test_tail_integral(q1_or_q2, tail_integral_d):
    with open(f"test_tail_integral_{tail_integral_d}d.yaml", 'r') as stream:
        args = SimpleNamespace(**yaml.safe_load(stream)['config'])

    args.q1_or_q2 = q1_or_q2

    args.output_folder = '/tmp'

    amci.train.train(args, _run=DummyClass(), _writer=DummyClass())


@pytest.mark.parametrize("q1_or_q2", ("q1", "q2"))
@pytest.mark.parametrize("factor", ("c0", "eps"))
def test_cancer(q1_or_q2, factor):
    with open(f"test_cancer.yaml", 'r') as stream:
        args = SimpleNamespace(**yaml.safe_load(stream)['config'])

    args.q1_or_q2 = q1_or_q2
    args.factor = factor

    args.output_folder = '/tmp'

    amci.train.train(args, _run=DummyClass(), _writer=DummyClass())


if __name__ == '__main__':
    pytest.main()
