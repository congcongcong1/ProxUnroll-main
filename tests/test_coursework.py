"""Check inverse-problem contracts, checkpoint loading, and no-GT equivalence."""

import numpy as np
import pytest
import torch
from coursework.baselines import adjoint, measure, fista_dct
from coursework.run_experiments import load_model


def test_adjoint_identity():
    rng = np.random.default_rng(4)
    h, w = rng.normal(size=(3, 8)), rng.normal(size=(4, 9))
    x, y = rng.normal(size=(8, 9)), rng.normal(size=(3, 4))
    np.testing.assert_allclose(np.sum(measure(x, h, w) * y), np.sum(x * adjoint(y, h, w)), atol=1e-10)


def test_fista_identity_operator_exact_soft_threshold():
    from scipy.fft import dctn, idctn
    rng = np.random.default_rng(0)
    x = rng.random((16, 16))
    c = dctn(x, norm='ortho')
    expected = idctn(np.sign(c) * np.maximum(np.abs(c) - .02, 0), norm='ortho')
    np.testing.assert_allclose(fista_dct(x, np.eye(16), np.eye(16), .02, 5), expected, atol=1e-10)


@pytest.mark.parametrize('solver,cr', [('hqs', .1), ('admm', .1), ('hqs', .5)])
def test_measurement_only_matches_original_forward(solver, cr):
    torch.set_num_threads(4)
    torch.manual_seed(123)
    model = load_model(solver, torch.device('cpu'))
    gt = torch.rand(1, 256, 256)
    with torch.inference_mode():
        h, w, _, wt = model.measurement_matrices(256, 256, cr)
        y = (h @ gt.unsqueeze(1) @ wt).squeeze(1)
        old = model(gt, cr)[0]
        new = model.reconstruct(y, (256, 256), cr)
        explicit = model.reconstruct(y, (256, 256), cr, sensing_matrices=(h[0, 0], w[0, 0]))
    if solver == 'admm':
        old = old[1:]
    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new, explicit, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        model.reconstruct(y, (256, 256), cr, stages=7)
    with pytest.raises(ValueError):
        model.measurement_matrices(256, 256, 0)
    with pytest.raises(ValueError):
        model.reconstruct(y[:, :-1], (256, 256), cr)
