"""Checks for causal pairing and equal loss scale in the matched training arms."""
import numpy as np
import pytest

from coursework.finetune_ablation import training_condition, stage_weights


def test_noise_switch_preserves_sampling_and_proposed_noise_draws():
    rates = [.01, .04, .1, .1, .1, .25, .25, .5]
    seen = set()
    for seed in [20261013, 20261014]:
        for step in range(800):
            on = training_condition(seed, step, rates, True)
            off = training_condition(seed, step, rates, False)
            assert on[0] == off[0]
            assert on[2] == off[2] == on[1]
            assert off[1] == 0
            seen.add(on[1])
    assert seen == {0., .01, .05}


def test_removing_intermediate_supervision_keeps_total_weight_one():
    weighted = stage_weights('weighted')
    final = stage_weights('final')
    assert len(weighted) == len(final) == 6
    assert sum(weighted) == pytest.approx(1.)
    assert sum(final) == pytest.approx(1.)
    assert final[:-1] == [0.] * 5
    # Identical RMSE at every stage must not change loss magnitude.
    losses = np.full(6, .123)
    assert np.dot(weighted, losses) == pytest.approx(np.dot(final, losses))


def test_noise_generation_matches_legacy_continuation_draws():
    rates = [.01, .04, .1, .1, .1, .25, .25, .5]
    for step in [6, 9999, 10000, 11999]:
        rng = np.random.default_rng(np.random.SeedSequence([20261010, step, 7]))
        legacy = (float(rng.choice(rates)), float(rng.choice([0., 0., .01, .05])))
        assert training_condition(20261010, step, rates, True)[:2] == legacy


def test_factorial_effects_use_original_units_and_keep_both_checkpoint_views(tmp_path):
    import csv
    import json
    from coursework.analyze_ablation import analyze

    rows = []
    values = {'hqs': .012, 'hqs_10000': .01, 'hqs_12000': .009, 'last_12000': .0095}
    for seed in [20261013, 20261014]:
        for view in ['best', 'last']:
            for noise, loss, mse in [('off', 'final', .012), ('on', 'final', .01),
                                     ('off', 'weighted', .011), ('on', 'weighted', .008)]:
                values[f'{view}_noise_{noise}_{loss}_s{seed}'] = mse
    for sequence, count, scale in [('long', 100, 1.), ('short', 1, 2.)]:
        for image in range(count):
            for method, mse in values.items():
                rows.append(dict(image=f'{sequence}_{image}', dataset='synthetic', sequence=sequence,
                    cr=.1, sigma=.05, seed=2026, method=method,
                    psnr=-10 * np.log10(mse * scale), ssim=.8, measurement_sha256='shared'))
    with (tmp_path / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / 'verification.json').write_text(json.dumps(dict(
        data_checkpoint_matrix_source_hashes_verified=True, rows=len(rows))))
    result = analyze(tmp_path)
    summary = next(r for r in result['summaries'] if r['method'] == 'best_noise_on_weighted_s20261013')
    assert summary['units'] == 2 and summary['images'] == 101
    assert summary['mse'] == pytest.approx(.008 * 1.5)
    for view in ['best', 'last']:
        effects = {r['effect']: r for r in result['factorial_effects']
                   if r['checkpoint_view'] == view and r['training_seed'] == 'mean_of_two'}
        assert effects['noise_main_mse_benefit']['mse_benefit'] == pytest.approx(.0025 * 1.5)
        assert effects['stage_main_mse_benefit']['mse_benefit'] == pytest.approx(.0015 * 1.5)
        assert effects['noise_stage_interaction_mse_benefit']['mse_benefit'] == pytest.approx(.001 * 1.5)
    assert result['negative_comparisons'] > 0
    assert len(result['seed_stability']) == 8
