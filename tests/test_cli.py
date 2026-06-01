import sys
import types

import pytest

from gatk_sv_gd import cli


def test_print_usage_lists_known_subcommands(capsys):
    cli._print_usage()

    captured = capsys.readouterr()
    assert "Usage: gatk-sv-gd <subcommand> [options]" in captured.out
    for name, description in cli.DESCRIPTIONS.items():
        assert name in captured.out
        assert description in captured.out
    assert captured.err == ""


def test_main_without_subcommand_prints_usage_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gatk-sv-gd"])

    with pytest.raises(SystemExit, match="1"):
        cli.main()

    captured = capsys.readouterr()
    assert "Usage: gatk-sv-gd <subcommand> [options]" in captured.out
    assert captured.err == ""


def test_main_help_prints_usage_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gatk-sv-gd", "--help"])

    with pytest.raises(SystemExit, match="0"):
        cli.main()

    captured = capsys.readouterr()
    assert "Subcommands:" in captured.out
    assert captured.err == ""


def test_main_unknown_subcommand_prints_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gatk-sv-gd", "unknown"])

    with pytest.raises(SystemExit, match="1"):
        cli.main()

    captured = capsys.readouterr()
    assert "Error: unknown subcommand 'unknown'" in captured.err
    assert "Usage: gatk-sv-gd <subcommand> [options]" in captured.err


def test_main_dispatches_to_submodule_and_rewrites_argv(monkeypatch):
    called = {}

    def _fake_main():
        called["argv"] = list(sys.argv)
        called["invoked"] = True

    def _fake_import_module(name):
        called["module_name"] = name
        return types.SimpleNamespace(main=_fake_main)

    monkeypatch.setattr(sys, "argv", ["gatk-sv-gd", "call", "--flag", "value"])
    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    cli.main()

    assert called == {
        "module_name": cli.SUBCOMMANDS["call"],
        "argv": ["gatk-sv-gd call", "--flag", "value"],
        "invoked": True,
    }