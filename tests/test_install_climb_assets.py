from __future__ import annotations

from pathlib import Path

from scripts.setup import install_climb_assets


def test_missing_collision_pins_lists_only_mismatched_packages(monkeypatch) -> None:
    versions = {"trimesh": "5.1.0", "coacd": None, "shapely": "2.0.0"}
    monkeypatch.setattr(
        install_climb_assets,
        "_installed_version",
        lambda name: versions[name],
    )

    assert install_climb_assets.missing_collision_pins() == [
        "coacd==1.0.14",
        "shapely==2.1.2",
    ]


def test_ensure_collision_build_skips_pip_when_pins_match(monkeypatch) -> None:
    monkeypatch.setattr(install_climb_assets, "missing_collision_pins", lambda: [])
    calls = []
    install_climb_assets.ensure_collision_build(run=lambda cmd: calls.append(cmd))
    assert calls == []


def test_ensure_collision_build_installs_only_missing_pins(monkeypatch) -> None:
    monkeypatch.setattr(
        install_climb_assets,
        "missing_collision_pins",
        lambda: ["coacd==1.0.14"],
    )
    calls = []
    install_climb_assets.ensure_collision_build(run=lambda cmd: calls.append(cmd))
    assert calls == [[install_climb_assets.sys.executable, "-m", "pip", "install", "coacd==1.0.14"]]


def test_install_downloads_robots_then_builds_collision(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(install_climb_assets, "ensure_collision_build", lambda: calls.append("deps"))

    import scripts.setup.download_assets as download_assets
    import scripts.setup.download_g1_collision as collision

    def download_all_hf(groups, cache):
        calls.append(("huggingface", tuple(groups), cache))

    monkeypatch.setattr(download_assets, "download_all_hf", download_all_hf)
    monkeypatch.setattr(download_assets, "download_all", lambda *_: calls.append("modelscope"))
    monkeypatch.setattr(collision, "build_assets", lambda: calls.append("collision"))

    cache = tmp_path / "cache"
    install_climb_assets.install_climb_assets(source="huggingface", cache_dir=cache)

    assert calls == ["deps", ("huggingface", ("robots",), cache), "collision"]


def test_default_source_is_modelscope_robots_only(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(install_climb_assets, "ensure_collision_build", lambda: None)

    import scripts.setup.download_assets as download_assets
    import scripts.setup.download_g1_collision as collision

    def download_all(groups, cache):
        calls.append((tuple(groups), cache))

    monkeypatch.setattr(download_assets, "download_all", download_all)
    monkeypatch.setattr(
        download_assets,
        "download_all_hf",
        lambda *_: (_ for _ in ()).throw(AssertionError("huggingface download should not run")),
    )
    monkeypatch.setattr(collision, "build_assets", lambda: calls.append("collision"))

    install_climb_assets.install_climb_assets(cache_dir=tmp_path)
    assert calls == [(("robots",), tmp_path), "collision"]


def test_main_forwards_source_and_skip_deps(monkeypatch) -> None:
    seen = {}

    def _install(*, source, cache_dir, skip_deps):
        seen["source"] = source
        seen["cache_dir"] = cache_dir
        seen["skip_deps"] = skip_deps

    monkeypatch.setattr(install_climb_assets, "install_climb_assets", _install)
    install_climb_assets.main(["--source", "huggingface", "--cache_dir", "cache", "--skip-deps"])
    assert seen == {
        "source": "huggingface",
        "cache_dir": Path("cache"),
        "skip_deps": True,
    }
