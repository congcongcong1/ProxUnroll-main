"""Accept proven recovery while preserving original nonzero exits."""
import json
from pathlib import Path
import time

from coursework.run_experiments import sha


def require_evaluation(base, phase):
    exit_code = int((base / 'logs' / f'evaluation_{phase}.exit').read_text())
    if exit_code == 0:
        return dict(original_exit_code=0, recovered=False)
    record = json.loads((base / f'evaluation_{phase}_recovery_complete.json').read_text())
    assert record['original_failed_exit_preserved'] == record['original_exit_code'] == exit_code
    assert int((base / 'logs' / f'recovery_{phase}.exit').read_text()) == 0
    archive = base / f'evaluation_{phase}_terminal_partial'
    for name, digest in record['partial_csv_hashes'].items():
        assert sha(archive / name) == digest
    directory = base / f'evaluation_{phase}'
    meta = json.loads((directory / 'provenance.json').read_text())
    assert meta['recovery']['partial_csv_hashes'] == record['partial_csv_hashes']
    original_names = {r['name'] for r in meta['data']}
    complete, missing = set(record['complete_originals_reused']), set(record['incomplete_originals_recomputed'])
    assert not complete & missing and complete | missing == original_names
    verification = json.loads((directory / 'verification.json').read_text())
    assert verification['rows'] == verification['float_arrays_recomputed'] == record['recovered_rows']
    assert verification['max_metric_error'] < 2e-5
    assert (directory / 'COMPLETE').exists()
    return dict(original_exit_code=exit_code, recovered=True, recovery_exit_code=0,
                reused_originals=len(complete), recovered_originals=len(missing),
                original_failure_and_partial_data_preserved=True)


def require_workflow(base):
    records = {phase: require_evaluation(base, phase) for phase in ['primary', 'regression']}
    original = int((base / 'logs/finalize.exit').read_text())
    assert original == 0 or any(r['recovered'] for r in records.values())
    return dict(evaluations=records, original_finalize_exit=original)


def finish_recovered_workflow(base):
    proof = require_workflow(base)
    for phase, record in proof['evaluations'].items():
        if not record['recovered']:
            continue
        directory = base / f'evaluation_{phase}'
        provenance = directory / 'provenance.json'
        meta = json.loads(provenance.read_text())
        recovery_meta_path = base / f'recovery_{phase}_1/inference/provenance.json'
        recovery_meta = json.loads(recovery_meta_path.read_text())
        assert recovery_meta['checkpoints'] == meta['checkpoints']
        assert recovery_meta['conditions'] == meta['conditions']
        meta['recovery']['hardware'] = {key: recovery_meta[key] for key in
                                      ['gpu', 'device', 'torch_version', 'cuda_version', 'platform']}
        meta['recovery']['inference_provenance_sha256'] = sha(recovery_meta_path)
        meta['timing'] += '; recovered originals use separately recorded hardware; do not pool timings as one-GPU benchmark'
        provenance.write_text(json.dumps(meta, indent=2) + '\n')
    original = base / 'workflow_status.json'
    saved = base / 'workflow_status_before_recovery.json'
    assert not saved.exists()
    saved.write_bytes(original.read_bytes())
    original.write_text(json.dumps(dict(phase='experiments_complete',
        completion_path='bounded missing-original recovery; original failed exits retained',
        proof=proof, last_checked_epoch=time.time()), indent=2) + '\n')


if __name__ == '__main__':
    from coursework.ablation_suite import BASE
    finish_recovered_workflow(BASE)
