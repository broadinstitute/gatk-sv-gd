"""Exercise VCF serialization and the complete pipeline outside conftest stubs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_integrate_with_real_htslib(tmp_path):
    dependency_check = subprocess.run(
        [sys.executable, "-c", "import pysam, pysam.bcftools, intervaltree"], capture_output=True, text=True, check=False,
    )
    if dependency_check.returncode:
        pytest.skip("Real pysam/bcftools and intervaltree are required for VCF integration checks")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, str(root / "tests/data/integrate_vcf_smoke.py"), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "passed").exists()
