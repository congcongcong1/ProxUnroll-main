"""A failed inference cannot become complete by changing its status file."""
import json
import pytest

from coursework.ablation_completion import require_evaluation


def test_nonzero_evaluation_requires_actual_recovery_exit(tmp_path):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/evaluation_regression.exit').write_text('1\n')
    with pytest.raises(FileNotFoundError):
        require_evaluation(tmp_path, 'regression')
    (tmp_path / 'evaluation_regression_recovery_complete.json').write_text(json.dumps(
        dict(original_failed_exit_preserved=1, original_exit_code=1)))
    with pytest.raises(FileNotFoundError):
        require_evaluation(tmp_path, 'regression')
    (tmp_path / 'logs/recovery_regression.exit').write_text('1\n')
    with pytest.raises(AssertionError):
        require_evaluation(tmp_path, 'regression')


def test_success_record_cannot_rewrite_the_original_failure(tmp_path):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/evaluation_regression.exit').write_text('1\n')
    (tmp_path / 'logs/recovery_regression.exit').write_text('0\n')
    (tmp_path / 'evaluation_regression_recovery_complete.json').write_text(json.dumps(
        dict(original_failed_exit_preserved=0, original_exit_code=0)))
    with pytest.raises(AssertionError):
        require_evaluation(tmp_path, 'regression')
