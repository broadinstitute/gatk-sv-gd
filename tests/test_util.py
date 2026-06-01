import logging
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd import _util


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_posterior_qual_helpers_handle_scalar_and_array_inputs():
    assert _util.posterior_probability_to_qual(1.0) == pytest.approx(99.0)
    assert _util.posterior_probability_to_qual(0.0) == pytest.approx(-99.0)

    arr = _util.posterior_probability_to_qual(np.array([0.1, 0.5, 0.9]), max_qual=20.0)
    assert arr.tolist() == pytest.approx([
        -10.0 * np.log10(9.0),
        0.0,
        10.0 * np.log10(9.0),
    ])

    called = _util.posterior_called_state_to_qual(
        np.array([0.9, 0.9, 0.1]),
        np.array([True, False, False]),
        max_qual=20.0,
    )
    assert called.tolist() == pytest.approx([
        10.0 * np.log10(9.0),
        -10.0 * np.log10(9.0),
        10.0 * np.log10(9.0),
    ])
    assert _util.posterior_called_state_to_qual(0.1, False, max_qual=20.0) == pytest.approx(
        10.0 * np.log10(9.0)
    )


def test_logging_stream_emits_only_warning_and_error_lines(monkeypatch):
    logger = logging.getLogger("gatk_sv_gd.teststream")
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    capture = _CaptureHandler()
    logger.addHandler(capture)
    session = SimpleNamespace(had_error=False)
    monkeypatch.setattr(_util, "_logging_session", session)

    stream = _util.LoggingStream(logger)
    try:
        assert stream.writable() is True
        assert stream.isatty() is False
        assert stream.write("plain status line\n") == len("plain status line\n")
        stream.write("WARNING: noisy but important\n")
        stream.write("ERROR: discarded\rERROR: failed loudly\n")
        stream.flush()
    finally:
        stream.close()
        logger.removeHandler(capture)

    assert [(record.levelno, record.getMessage()) for record in capture.records] == [
        (logging.WARNING, "WARNING: noisy but important"),
        (logging.ERROR, "ERROR: failed loudly"),
    ]
    assert session.had_error is True
    assert stream.write("ignored") == 0


def test_privacy_helpers_sanitize_paths_and_sensitive_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "gatk-sv-gd",
        "plot",
        "--output-dir=/tmp/out",
        "--flag",
    ])

    args = SimpleNamespace(
        output_dir="/tmp/out",
        sample_id="S1",
        threads=4,
        labels=["A", "B"],
        dry_run=True,
        note="safe-value",
    )
    invocation = _util._privacy_safe_invocation(args)

    assert invocation["argument_count"] == 3
    assert invocation["flags"] == ["--flag", "--output-dir"]
    assert invocation["parsed_args"]["output_dir"] == {"provided": True}
    assert invocation["parsed_args"]["sample_id"] == {"provided": True}
    assert invocation["parsed_args"]["threads"] == 4
    assert invocation["parsed_args"]["labels"] == {"count": 2}
    assert invocation["parsed_args"]["note"] == "safe-value"

    assert _util._sanitize_log_text("see /tmp/private/file.tsv and report.txt") == "see <path> and <file>"
    assert _util._sanitize_log_fields({"path": "/tmp/private/file.tsv", "nested": ("report.txt", 1)}) == {
        "path": "<path>",
        "nested": ["<file>", 1],
    }
    assert _util._private_value_summary(["x", "y"]) == {"count": 2, "provided": True}
    assert _util._private_value_summary({"x": 1}) == {"count": 1, "provided": True}
    assert _util._is_sensitive_arg_key("raw_counts") is True
    assert _util._is_sensitive_arg_key("threads") is False
    assert "<path>" in _util._json_dumps({"path": "/tmp/private/file.tsv"})
    assert _util._json_safe({"obj": SimpleNamespace(a=1)}) == {"obj": "namespace(a=1)"}


def test_runtime_helpers_cover_git_dependency_environment_and_handlers(monkeypatch):
    managed = logging.NullHandler()
    unmanaged = logging.NullHandler()
    setattr(managed, _util._MANAGED_HANDLER_ATTR, True)

    logger = logging.getLogger("gatk_sv_gd.remove-test")
    logger.handlers = [managed, unmanaged]
    _util._remove_managed_handlers(logger)
    assert logger.handlers == [unmanaged]

    assert _util.get_logger().name == "gatk_sv_gd"
    assert _util.get_logger("child").name == "gatk_sv_gd.child"
    assert _util.get_logger("gatk_sv_gd.child").name == "gatk_sv_gd.child"
    assert _util._infer_command_name() == "gatk-sv-gd"

    monkeypatch.setattr(
        _util.importlib_metadata,
        "version",
        lambda name: "1.2.3" if name == "numpy" else (_ for _ in ()).throw(_util.importlib_metadata.PackageNotFoundError()),
    )
    assert _util._dependency_version("numpy") == "1.2.3"
    assert _util._dependency_version("missing") == "unknown"

    class _Result:
        returncode = 0
        stdout = "abc123\n"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Result())
    assert _util._run_git_command("/tmp", ["rev-parse", "HEAD"]) == "abc123"

    def raise_os_error(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(subprocess, "run", raise_os_error)
    assert _util._run_git_command("/tmp", ["status", "--porcelain"]) is None

    monkeypatch.setattr(_util.platform, "platform", lambda: "TestOS-1.0")
    monkeypatch.setattr(_util.os, "cpu_count", lambda: 8)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    env = _util._environment_metadata()
    assert env == {
        "platform": "TestOS-1.0",
        "cpu_count": 8,
        "env": {
            "OMP_NUM_THREADS": "2",
            "PYTHONHASHSEED": "0",
        },
    }


def test_get_sample_columns_excludes_metadata_columns():
    df = pd.DataFrame(
        {
            "Chr": ["chr1"],
            "Start": [0],
            "End": [100],
            "source_file": ["raw"],
            "S1": [1.0],
            "S2": [2.0],
        }
    )

    assert _util.get_sample_columns(df) == ["S1", "S2"]
