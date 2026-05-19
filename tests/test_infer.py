import sys

import pytest

from gatk_sv_gd.infer import parse_args


def test_parse_args_uses_learned_baf_temperature_by_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gatk-sv-gd infer", "--preprocessed-dir", "./preprocess", "-o", "./out"],
    )

    args = parse_args()

    assert args.fixed_baf_temperature is False


def test_parse_args_allows_explicit_fixed_baf_temperature(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd infer",
            "--preprocessed-dir",
            "./preprocess",
            "-o",
            "./out",
            "--fixed-baf-temperature",
        ],
    )

    args = parse_args()

    assert args.fixed_baf_temperature is True