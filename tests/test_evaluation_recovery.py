"""Recovery must preserve a complete original rather than accepting a count."""
from coursework.recover_ablation_evaluation import complete_originals


def test_recovery_partition_rejects_duplicate_or_missing_conditions():
    meta = dict(data=[dict(name=n) for n in ['complete', 'partial', 'duplicate']],
                arguments=dict(methods='hqs,new'), conditions=[[.1, 0, 2026], [.1, .05, 2027]])
    rows = [dict(image=n, cr=str(cr), sigma=str(sigma), seed=str(seed), method=method)
            for n in ['complete', 'partial', 'duplicate']
            for cr, sigma, seed in meta['conditions'] for method in ['hqs', 'new']]
    rows = [r for r in rows if not (r['image'] == 'partial' and r['sigma'] == '0.05' and r['method'] == 'new')]
    duplicate = [r for r in rows if r['image'] == 'duplicate']
    rows.remove(duplicate[-1])
    rows.append(dict(duplicate[0]))
    assert complete_originals(rows, meta) == {'complete'}


def test_recovery_partition_normalizes_equivalent_numeric_strings():
    meta = dict(data=[dict(name='x')], arguments=dict(methods='hqs'), conditions=[[.1, 0, 2026]])
    rows = [dict(image='x', cr='0.100000', sigma='0.0', seed='2026', method='hqs')]
    assert complete_originals(rows, meta) == {'x'}
