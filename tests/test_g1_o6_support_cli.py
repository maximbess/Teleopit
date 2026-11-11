import json
import sys

import pytest


def test_transfer_cli_exposes_explicit_side_timing_and_cycles(monkeypatch, capsys):
    from teleopit.sim.g1_o6_support import __main__, transfer

    monkeypatch.setattr(
        transfer,
        "run_transfer",
        lambda book, output, **kw: {"book": book, "output": output, **kw},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "o6-support",
            "transfer",
            "--evidence",
            "book.json",
            "--output",
            "trial",
            "--mode",
            "reacquire",
            "--side",
            "right",
            "--cycles",
            "3",
            "--alternating",
            "--three-seconds",
            "10",
            "--physics-dt",
            "0.0005",
        ],
    )
    __main__.main()
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "reacquire" and result["side"] == "right"
    assert result["cycles"] == 3 and result["alternating"]
    assert result["three_seconds"] == 10 and result["physics_dt"] == 0.0005


def test_invalid_cli_experiment_cannot_start(monkeypatch):
    from teleopit.sim.g1_o6_support import __main__

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "o6-support",
            "transfer",
            "--evidence",
            "book",
            "--output",
            "trial",
            "--target-share",
            "0.9",
        ],
    )
    with pytest.raises(SystemExit) as error:
        __main__.main()
    assert error.value.code == 2
