from bodoai_examples import simple


def test_simple():
    pi = simple.calc_pi(100000)
    assert 3.1 <= pi < 3.2
