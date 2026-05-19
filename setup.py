from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="gatk-sv-gd",
    version="0.1.0",
    description="Genomic Disorder CNV detection from binned read counts",
    long_description=(Path(__file__).parent / "README.md").read_text(
        encoding="utf-8"
    ),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    license="BSD-3-Clause",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "pysam",
        "torch",
        "pyro-ppl",
        "tqdm",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "gatk-sv-gd=gatk_sv_gd.cli:main",
        ],
    },
)
