import json
import logging
import re
import sys
import warnings
from argparse import Namespace

from gatk_sv_gd import _util


ISO_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z "
)


def test_setup_logging_routes_diagnostics_off_stdout(tmp_path, capfd):
    session = _util.setup_logging(
        str(tmp_path),
        filename="test.log",
        command="unit-test",
        args=Namespace(
            input_path="/sensitive/project/input.tsv",
            output_dir=str(tmp_path),
            sample_id="sensitive_sample_id",
            calling_mode="posterior-marginal",
        ),
        seed_info={"numpy": 123},
    )
    try:
        print("plain diagnostic")
        print("WARNING: suspicious input")
        sys.stderr.write("ERROR: failed optional step\n")
        _util.vlog("debug detail hidden at info level")
        session.flush()
    finally:
        session.close()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert "plain diagnostic" not in captured.err
    assert "Starting gatk-sv-gd unit-test" in captured.err
    assert "WARNING gatk_sv_gd.stdout: WARNING: suspicious input" in captured.err
    assert "ERROR gatk_sv_gd.stderr: ERROR: failed optional step" in captured.err

    log_text = (tmp_path / "test.log").read_text()
    log_lines = log_text.splitlines()
    assert all(ISO_TIMESTAMP_RE.match(line) for line in log_lines)
    assert "Starting gatk-sv-gd unit-test" in log_text
    assert "plain diagnostic" not in log_text
    assert "WARNING: suspicious input" in log_text
    assert "ERROR: failed optional step" in log_text
    assert "debug detail hidden" not in log_text
    assert "/sensitive/project" not in log_text
    assert "input.tsv" not in log_text
    assert "sensitive_sample_id" not in log_text


def test_json_logging_writes_parseable_records(tmp_path):
    session = _util.setup_logging(
        str(tmp_path),
        filename="test.jsonl",
        command="json-test",
        args=Namespace(output_dir=str(tmp_path)),
        seed_info={"numpy.default_rng": 42},
        log_format="json",
    )
    try:
        print("json diagnostic")
        session.flush()
    finally:
        session.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "test.jsonl").read_text().splitlines()
    ]
    messages = [record["message"] for record in records]
    assert messages[0] == "Starting gatk-sv-gd json-test"
    assert any(message.startswith("Random seeds:") for message in messages)
    assert any("completed successfully" in message for message in messages)
    assert "json diagnostic" not in messages
    assert all(record["level"] == "INFO" for record in records)


def test_setup_logging_suppresses_torch_tracer_warnings(tmp_path, capfd, caplog):
    import torch

    caplog.set_level(logging.WARNING, logger="py.warnings")
    session = _util.setup_logging(
        str(tmp_path),
        filename="test.log",
        command="trace-warning-test",
        args=Namespace(output_dir=str(tmp_path)),
    )
    try:
        warnings.warn("expected trace compile noise", torch.jit.TracerWarning)
        warnings.warn("ordinary warning still visible", UserWarning)
        session.flush()
    finally:
        session.close()

    captured = capfd.readouterr()
    log_text = (tmp_path / "test.log").read_text()
    assert "expected trace compile noise" not in captured.err
    assert "expected trace compile noise" not in log_text
    warning_messages = [record.getMessage() for record in caplog.records]
    assert all("expected trace compile noise" not in message for message in warning_messages)
    assert any("ordinary warning still visible" in message for message in warning_messages)