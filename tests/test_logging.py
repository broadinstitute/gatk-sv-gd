import json
import re
import sys
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
        args=Namespace(input_path="input.tsv", output_dir=str(tmp_path)),
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
    assert "INFO gatk_sv_gd.stdout: plain diagnostic" in captured.err
    assert "WARNING gatk_sv_gd.stdout: WARNING: suspicious input" in captured.err
    assert "ERROR gatk_sv_gd.stderr: ERROR: failed optional step" in captured.err

    log_lines = (tmp_path / "test.log").read_text().splitlines()
    assert all(ISO_TIMESTAMP_RE.match(line) for line in log_lines)
    assert any("run_start" in line and '"command": "unit-test"' in line for line in log_lines)
    assert any("plain diagnostic" in line for line in log_lines)
    assert any("WARNING: suspicious input" in line for line in log_lines)
    assert any("ERROR: failed optional step" in line for line in log_lines)
    assert not any("debug detail hidden" in line for line in log_lines)


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
    assert records[0]["event"] == "run_start"
    assert records[0]["command"] == "json-test"
    assert records[0]["random_seeds"] == {"numpy.default_rng": 42}
    assert records[-1]["message"] == "json diagnostic"
    assert records[-1]["level"] == "INFO"
    assert records[-1]["logger"] == "gatk_sv_gd.stdout"