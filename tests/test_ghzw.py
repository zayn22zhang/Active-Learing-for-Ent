import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from al4qed.states import ghz_w_3x3, isotropic_state
from al4qed.dataset import sample_state, validate_density, reduced_states
from al4qed.oracle import ppt_visibility


def test_endpoints_and_partition():
    np.testing.assert_allclose(ghz_w_3x3(0,0), np.eye(9)/9)
    np.testing.assert_allclose(ghz_w_3x3(1,0), isotropic_state(1,3))
    a,b = reduced_states(ghz_w_3x3(0,1),(3,3))
    np.testing.assert_allclose(a,np.diag([.5,.5,0]))
    np.testing.assert_allclose(a,b)
    assert ppt_visibility(ghz_w_3x3(1,0),(3,3))[0] == pytest.approx(.25)
    assert ppt_visibility(ghz_w_3x3(0,1),(3,3))[0] == pytest.approx(2/11)


@pytest.mark.parametrize('p,q', [(-.1,.2),(.5,.6),(float('nan'),0),(0,float('inf'))])
def test_invalid_weights(p,q):
    with pytest.raises(ValueError): ghz_w_3x3(p,q)


def test_simplex_physical_sampling():
    rng=np.random.default_rng(91)
    for _ in range(100):
        validate_density(sample_state((3,3),rng,'ghz_w_3x3'),(3,3))
    with pytest.raises(ValueError):sample_state((2,2),rng,'ghz_w_3x3')
    np.testing.assert_allclose(sample_state((3,3),np.random.default_rng(7),'ghz_w_3x3'),
                               sample_state((3,3),np.random.default_rng(7),'ghz_w_3x3'))


def test_cli_accepts_family():
    from al4qed.train import parser
    a=parser().parse_args(['--distribution','ghz_w_3x3','--dims','3','3'])
    assert a.distribution=='ghz_w_3x3'
