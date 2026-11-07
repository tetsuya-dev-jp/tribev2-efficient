import pytest

from benchmarks import evaluate_video_cohort as cohort


def test_cohort_discovery_rejects_a_missing_movie_instead_of_dropping_subject(tmp_path, monkeypatch):
    monkeypatch.setattr(cohort, "N_SUBJECTS", 2)
    for subject in ("example_a", "example_b"):
        for run in cohort.RUNS:
            (tmp_path / f"bold_fs5_{subject}_{run}.npz").touch()
    subjects, files = cohort.discover(tmp_path)
    assert len(subjects) == 2
    files["MOVIE2_HO1"]["example_b"].unlink()
    with pytest.raises(AssertionError, match="Subject sets differ"):
        cohort.discover(tmp_path)


def test_cohort_summary_keeps_worst_subject_when_signed_deltas_cancel():
    rows = [{"run": "MOVIE1_CC1", "baseline_mean_parcel_r": .2,
             "candidate_mean_parcel_r": .2 + delta, "mean_parcel_r_delta": delta,
             "max_abs_parcel_r_delta": .002, "relative_mse_change": delta,
             "passed": False} for delta in (-.0002, .0002)]
    summary = cohort.summarize(rows)[0]
    assert summary["mean_parcel_r_delta"] == 0
    assert summary["max_abs_subject_mean_r_delta"] == .0002
    assert summary["max_abs_parcel_r_delta"] == .002
    assert summary["n_failed_subjects"] == 2
    assert summary["subject_mean_r_delta_quantiles"][0] == -.0002
    assert summary["subject_mean_r_delta_quantiles"][-1] == .0002
