#!/usr/bin/env python3
"""Assemble the GNU Octave package tarball for L-UQ.

    python octave/build_octave_package.py

Produces octave/l-uq-<version>.tar.gz, laid out as GNU Octave's `pkg`
expects:

    l-uq-<version>/
        DESCRIPTION
        COPYING
        INDEX
        NEWS
        inst/*.m
        inst/test/*.m

The .m sources are copied from the repository root rather than
duplicated, so the MATLAB toolbox and the Octave package are built from
one set of files and cannot drift apart. The version is read from
DESCRIPTION.

Install and check the result with:

    octave --eval "pkg install octave/l-uq-2.0.0.tar.gz"
    octave --eval "pkg load l-uq; run('tests/octave_verify.m')"

This is a Python script rather than a .m one so that the package can be
assembled on a machine without Octave installed; nothing in it needs
Octave to run.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The .m sources that make up the package, all from the repository root.
SOURCES = [
    "lmom.m",
    "LegendreShiftPoly.m",
    "Identify_dist.m",
    "Identify_dist_bootstrap.m",
    "Parameter_estimation.m",
    "parameter_identify.m",
    "fit_best.m",
    "PDF_l.m",
    "CDF_l.m",
    "Random_l.m",
    "JSDiv.m",
    "KLDiv.m",
    "KSStat.m",
    "luq_dist.m",
    "luq_bin_fit.m",
    "luq_percentile.m",
    "demo_octave.m",
]

# Test scripts, shipped inside the package so `pkg test` style checks and
# manual verification runs are available after installation.
TEST_SOURCES = ["tests/octave_verify.m"]

# Files that are deliberately NOT packaged:
#   package_toolbox.m      - builds the MATLAB .mltbx, irrelevant to Octave
#   demo_example.m         - the richer MATLAB demo, which needs fitdist,
#                            histogram and ecdf; demo_octave.m is the
#                            portable equivalent shipped in its place
#   tests/test_uq_matlab.m - uses MATLAB's unittest framework, which
#                            Octave does not provide (octave_verify.m is
#                            the Octave equivalent)


def read_version(description: Path) -> str:
    for line in description.read_text(encoding="utf-8").splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"no Version: field in {description}")


def main() -> None:
    description = HERE / "DESCRIPTION"
    version = read_version(description)
    pkgname = f"l-uq-{version}"

    missing = [name for name in SOURCES + TEST_SOURCES
               if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing sources: " + ", ".join(missing))

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / pkgname
        (staging / "inst" / "test").mkdir(parents=True)

        shutil.copy2(description, staging / "DESCRIPTION")
        shutil.copy2(HERE / "INDEX", staging / "INDEX")
        shutil.copy2(HERE / "NEWS", staging / "NEWS")
        # Octave's pkg expects the licence as COPYING.
        shutil.copy2(ROOT / "LICENSE", staging / "COPYING")

        for name in SOURCES:
            shutil.copy2(ROOT / name, staging / "inst" / Path(name).name)
        for name in TEST_SOURCES:
            shutil.copy2(ROOT / name, staging / "inst" / "test" / Path(name).name)

        out = HERE / f"{pkgname}.tar.gz"
        with tarfile.open(out, "w:gz") as tar:
            tar.add(staging, arcname=pkgname)

    n_files = len(SOURCES) + len(TEST_SOURCES)
    print(f"Wrote {out} ({n_files} .m files, version {version})")
    print(f"Install with:  octave --eval \"pkg install {out.name}\"")


if __name__ == "__main__":
    main()
