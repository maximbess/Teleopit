"""A partial advisory report cache must never block campaign resumption."""


def test_interrupted_cache_is_recomputed_and_complete_cache_is_reused(
    tmp_path, monkeypatch
):
    from teleopit.sim.g1_o6_support import pose_results

    run = tmp_path / "runs/example"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    cache = tmp_path / "comparison_cache/example.json"
    cache.parent.mkdir()
    cache.write_text('{"source_and_analysis_sha256":')
    calls = []

    def measured_summary(*args):
        calls.append(1)
        return {"candidate": "example", "strict": "FAIL"}

    monkeypatch.setattr(pose_results, "summarize_record", measured_summary)
    record = {"id": "example", "output": "runs/example"}
    expected = {"candidate": "example", "strict": "FAIL"}
    assert pose_results.cached_summary(tmp_path, record) == expected
    assert pose_results.cached_summary(tmp_path, record) == expected
    assert calls == [1]
    assert list(cache.parent.glob("*.tmp")) == []
