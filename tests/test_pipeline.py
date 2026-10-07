import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'al4qed'))
from dataset import reduced_states, extract_features_advanced, random_density_matrix, validate_density
from oracle import AdaptivePolytopeOracle, OracleLabel
from states import isotropic_state, horodecki_3x3
from active_learning import ActiveLearner
from acquisition import get_acquisition_strategy
from train import get_model, parser, run_experiment


def test_partial_trace_rectangular_complex_product():
    v = np.array([1,1j])/np.sqrt(2)
    a = np.outer(v,v.conj()); b = np.diag([.1,.3,.6])
    rho = np.kron(a,b)
    ar,br = reduced_states(rho,(2,3))
    np.testing.assert_allclose(ar,a)
    np.testing.assert_allclose(br,b)
    assert len(extract_features_advanced(rho,(2,3))) == 2*36+6+1+2+3


def test_invalid_density_and_implicit_dims():
    with pytest.raises(ValueError):
        extract_features_advanced(np.eye(6)/6)
    with pytest.raises(ValueError):
        validate_density(np.diag([1.2,-.2]), (1,2))
    with pytest.raises(ValueError):
        isotropic_state(1.1)


@pytest.mark.parametrize('p,expected',[(0,1),(0.2,1),(1,1/3),(.5,2/3)])
def test_exact_two_qubit_visibility(p,expected):
    r = AdaptivePolytopeOracle(backend='ppt').query(isotropic_state(p),2,2)
    assert r['chi'] == pytest.approx(expected)
    assert r['label'] == (OracleLabel.ENTANGLED if p>1/3 else OracleLabel.SEPARABLE)


def test_ppt_is_not_exact_in_3x3():
    with pytest.raises(ValueError):
        AdaptivePolytopeOracle(backend='ppt').query(horodecki_3x3(.5),3,3)


def test_low_inner_estimate_never_proves_entanglement(monkeypatch):
    stub = types.ModuleType('stub_provider')
    stub.query = lambda *a, **k: dict(chi=.4,status='validated',target_kind='inner_polytope_lower_estimate')
    monkeypatch.setitem(sys.modules,'stub_provider',stub)
    oracle = AdaptivePolytopeOracle(backend='external',external='stub_provider:query',backend_version='test-v1')
    assert oracle.query(np.eye(9)/9,3,3)['label'] == OracleLabel.UNKNOWN
    assert oracle.query(horodecki_3x3(.5),3,3)['label'] == OracleLabel.UNKNOWN


@pytest.mark.parametrize('raw', [dict(chi=0,status='failed'),dict(chi=np.nan,status='validated',target_kind='inner_polytope_lower_estimate')])
def test_failed_oracle_not_a_label(monkeypatch,raw):
    stub=types.ModuleType('stub_provider'); stub.query=lambda *a,**k: raw
    monkeypatch.setitem(sys.modules,'stub_provider',stub)
    oracle=AdaptivePolytopeOracle(backend='external',external='stub_provider:query',backend_version='test-v1')
    with pytest.raises(RuntimeError):
        oracle.query(np.eye(9)/9,3,3)


def test_cache_key_includes_configuration(tmp_path):
    rho=isotropic_state(.8)
    a=AdaptivePolytopeOracle(backend='ppt', cache_dir=tmp_path)
    first=a.query(rho,2,2); second=a.query(rho,2,2)
    assert not first['cache_hit'] and second['cache_hit']
    assert a.backend_calls == 1 and a.requests == 2
    b=AdaptivePolytopeOracle(backend='ppt', cache_dir=tmp_path,seed=17)
    assert not b.query(rho,2,2)['cache_hit']


def learner(seed=10, validation_data=None):
    return ActiveLearner(AdaptivePolytopeOracle(backend='ppt'),get_model('rf',seed=seed),seed=seed,
                         validation_data=validation_data)


def test_scaler_no_validation_leakage_and_no_prediction_refit():
    a=learner(validation_data=(np.array([[1000.,1000.],[2000.,2000.]]),np.array([.8,.9])))
    a.add_labeled([[0.,1.],[2.,3.],[4.,5.]],[.3,.5,.8]); a.train_surrogate()
    np.testing.assert_allclose(a.scaler.mean_,[2,3])
    before=a.predict_chi([1,2]); mean=a.scaler.mean_.copy()
    a.add_labeled([500,500],.9)
    assert a.predict_chi([1,2]) == before
    np.testing.assert_array_equal(a.scaler.mean_,mean)
    a.train_surrogate()
    assert a.trained_count == 4


@pytest.mark.parametrize('strategy',['random','uncertainty','boundary','margin','hybrid'])
def test_acquisition_empty_and_exhaustion(strategy):
    a=learner(); rng=np.random.default_rng(1)
    rhos=[random_density_matrix(4,rng) for _ in range(6)]
    X=[extract_features_advanced(r,(2,2)) for r in rhos]
    a.add_labeled(X,[.1,.2,.3,.4,.5,.6]); a.train_surrogate()
    empty=get_acquisition_strategy(strategy)(a,[],10)
    assert len(empty[0])==0
    a.add_to_pool(rhos[:2],[(2,2)]*2)
    assert len(a.query_uncertain(10,strategy=strategy))==2
    assert not a.pool
    assert len(a.query_uncertain(1,strategy=strategy))==0


def test_score_alignment_and_last_retraining():
    a=learner(); rng=np.random.default_rng(11)
    states=[random_density_matrix(4,rng) for _ in range(7)]
    X=[extract_features_advanced(r,(2,2)) for r in states]
    a.add_labeled(X[:4],[.2,.4,.6,.8]); a.train_surrogate()
    a.add_to_pool(states[4:],[(2,2)]*3)
    indices,scores=get_acquisition_strategy('margin')(a,a.pool,2)
    expected=[(a.pool[int(i)]['sample_id'],s) for i,s in zip(indices,scores)]
    a.query_uncertain(2,strategy='margin')
    assert [(r['sample_id'],r['score']) for r in a.query_history]==expected
    a.active_learning_cycle(1,1,strategy='random',verbose=False)
    assert a.trained_count==len(a.y_labeled)==7


def test_paired_end_to_end_and_checkpoints(tmp_path):
    args=parser().parse_args(['--backend','ppt','--dims','2','2','--model','rf',
        '--initial','8','--pool','3','--queries','10','--cycles','2','--val','4',
        '--test','8','--seeds','0','--strategies','random','uncertainty','--no-plot',
        '--output',str(tmp_path/'run')])
    rows=run_experiment(args)
    assert len(rows)==4
    assert rows[0]['mae']==rows[2]['mae']
    assert [r['train_queries'] for r in rows]==[8,11,8,11]
    assert all(r['model_trained_on']==r['train_queries'] for r in rows)
    assert all(r['total_logical_queries']==r['train_queries']+12 for r in rows)
    import joblib
    model=joblib.load(tmp_path/'run/rf_random_seed0/model.joblib')
    data=np.load(tmp_path/'run/test_set.npz')
    pred=np.clip(model['model'].predict(model['scaler'].transform(data['X'])),0,1)
    saved=np.load(tmp_path/'run/rf_random_seed0/predictions.npz')['prediction']
    np.testing.assert_allclose(pred,saved)


def test_single_sample_mc_dropout_preserves_batchnorm():
    torch=pytest.importorskip('torch')
    from network import dropout_only, ChiPredictor
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.BatchNorm1d(8),
                              torch.nn.Dropout(.4),torch.nn.Linear(8,1))
    model.train(); before=model[1].running_mean.clone()
    with dropout_only(model):
        results=[model(torch.ones(1,4)).detach() for _ in range(20)]
    assert model.training and model[1].training
    assert torch.equal(before,model[1].running_mean)
    assert torch.stack(results).std()>0
    net=ChiPredictor(4); net.train()
    mean,std=net.predict_with_uncertainty(torch.ones(1,4))
    assert mean.shape==std.shape==(1,) and net.training


def test_neural_remainder_one_and_final_fit():
    torch=pytest.importorskip('torch'); torch.set_num_threads(1)
    from network import ChiPredictor
    a=ActiveLearner(AdaptivePolytopeOracle(backend='ppt'),ChiPredictor(4),batch_size=4,n_epochs=3,
                    validation_data=(np.ones((2,4),np.float32),[.3,.4]))
    a.add_labeled(np.random.default_rng(0).normal(size=(9,4)),np.linspace(.2,.8,9))
    a.train_surrogate()
    assert a.trained_count==9
    assert np.isfinite(a.predict_uncertainty(np.ones(4))).all()


def test_local_sdp_bell_and_maximally_mixed():
    pytest.importorskip('cvxpy')
    a=AdaptivePolytopeOracle(N=12,max_iter=3,seed=21)
    assert a.query(isotropic_state(0),2,2)['chi']==pytest.approx(1,abs=1e-5)
    bell=a.query(isotropic_state(1),2,2)
    # A finite inner approximation need not attain the true boundary.
    assert 0 < bell['chi'] <= 1/3+1e-5
    from adaptive import bipartite_visibility_sdp
    vectors=[np.array(v,dtype=complex)/np.linalg.norm(v) for v in
             ([1,0],[0,1],[1,1],[1,-1],[1,1j],[1,-1j])]
    vertices=[np.outer(v,v.conj()) for v in vectors]
    chi,_=bipartite_visibility_sdp(isotropic_state(1),vertices,2)
    assert chi==pytest.approx(1/3,abs=1e-5)


def test_balanced_initial_polytope():
    pytest.importorskip('cvxpy')
    from adaptive import random_inner_polytope
    vertices=random_inner_polytope(3,18,np.random.default_rng(0))
    np.testing.assert_allclose(np.mean(vertices,axis=0),np.eye(3)/3,atol=1e-14)
    for vertex in vertices:
        validate_density(vertex)


def test_external_bridge_version_guard():
    from ohst_bridge import BRIDGE_VERSION
    with pytest.raises(ValueError):
        AdaptivePolytopeOracle(backend='external',external='ohst_bridge:query',backend_version='wrong')


def test_package_import():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import al4qed
    assert al4qed.AdaptivePolytopeOracle(backend='ppt').get_chi(isotropic_state(1),2,2)==pytest.approx(1/3)
