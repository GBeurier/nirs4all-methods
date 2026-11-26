# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Compare ReflectanceFromReferences with reflectance computed by vendor software.

Spectrometer files of AvaSoft (Avantes), OceanView (Ocean Optics), SVC and
Spectral Evolution software store the raw sample, white (or reference) and
dark spectra together with the reflectance or transmittance the vendor software
computed from them. This script reads those files with nirs4all-formats, runs
n4m's ReflectanceFromReferences on the raw spectra and reports how far the
vendor values are. The files are not copied into this repository; the report
is ``parity/results/spectral_conversions_vendor.json``.

Spectral Evolution DARWin additionally applies its own detector splice and
overlap processing, so its reflectance differs from the plain ratio by a
per-detector factor; SpliceCorrection reduces, but does not remove, that
difference (the procedure is not documented by the vendor).

Usage::

    N4M_LIB_PATH=... PYTHONPATH=bindings/python/src \
    python parity/spectral_conversions/vendor_check.py --formats-samples ../nirs4all-formats/samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "parity" / "results" / "spectral_conversions_vendor.json"

# (file, software, sample, white, dark or None, vendor output)
FILES = [
    (
        "avantes/avantes2.TRM",
        "AvaSoft",
        "sample",
        "white_reference",
        "dark_reference",
        "transmittance",
    ),
    (
        "avantes/avantes_trans.TRM",
        "AvaSoft",
        "sample",
        "white_reference",
        "dark_reference",
        "transmittance",
    ),
    (
        "avantes/avantes_export_long.ttt",
        "AvaSoft export",
        "sample",
        "ref",
        "dark",
        "transmittance",
    ),
    (
        "avantes/avasoft8.txt",
        "AvaSoft 8 export",
        "sample",
        "reference",
        "dark",
        "reflectance",
    ),
    (
        "ocean_optics/OceanOptics_Linux.ProcSpec",
        "OceanView",
        "sample",
        "white_reference",
        "dark_reference",
        "transmittance",
    ),
    (
        "ocean_optics/OceanOptics_Windows.ProcSpec",
        "OceanView",
        "sample",
        "white_reference",
        "dark_reference",
        "transmittance",
    ),
    (
        "ocean_optics/OceanOptics_period.jdx",
        "OceanView JCAMP",
        "sample",
        "white_reference",
        "dark_reference",
        "processed",
    ),
    (
        "ocean_optics/whiteref.ProcSpec",
        "OceanView",
        "sample",
        "white_reference",
        "dark_reference",
        "reflectance",
    ),
    (
        "spectral_evolution/1566060_09506_working.sed",
        "Spectral Evolution DARWin",
        "norm__dn_target",
        "norm__dn_ref_",
        None,
        "reflect__%",
    ),
    (
        "spectral_evolution/serbinsh_cvars_grape_leaf.sed",
        "Spectral Evolution DARWin",
        "norm__dn_target",
        "norm__dn_ref_",
        None,
        "reflect__1_0",
    ),
] + [
    (f"svc_ger/{name}", "SVC HR", "target", "reference", None, "reflectance")
    for name in (
        "ACPL_D2_P1_B_1_001.sig",
        "ACPL_D2_P1_B_2_001.sig",
        "ACPL_D2_P1_M_1_000.sig",
        "ACPL_D2_P1_M_2_000.sig",
        "ACPL_D2_P1_T_1_000.sig",
        "ACPL_D2_P1_T_1_WR_000.sig",
        "ACPL_D2_P1_T_2_000.sig",
        "ACPL_F3_P2_B_1_000.sig",
        "BNL13001_000_laptop.sig",
        "BNL13001_000_moc.sig",
        "BNL13002_000_laptop.sig",
        "serbinsh_BEO_CakeEater_Pheno_026_resamp.sig",
        "serbinsh_gr070214_003.sig",
    )
]


def main() -> int:
    import nirs4all_formats
    from n4m import roles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats-samples", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, software, sample, white, dark, vendor in FILES:
        record = nirs4all_formats.open_records(args.formats_samples / name)[0]
        signals = record["signals"]
        values = {
            key: np.asarray(signals[key]["values"], dtype=float)
            for key in (sample, white, dark, vendor)
            if key is not None
        }
        unit = signals[vendor].get("unit") or ""
        percent = "%" in unit
        params = {"white_reference": values[white].tolist(), "as_percent": percent}
        if dark is not None:
            params["dark_reference"] = values[dark].tolist()
        X = values[sample][None, :]
        ours = roles.ReflectanceFromReferences(**params).fit(X).transform(X)[0]
        theirs = values[vendor]
        valid = np.isfinite(ours) & np.isfinite(theirs)
        # Bands whose reference is at noise level (white - dark below 1 % of its
        # maximum) carry no reflectance; vendors print arbitrary values there.
        span = values[white] - (values[dark] if dark is not None else 0.0)
        signal = valid & (span >= 0.01 * np.nanmax(span))
        diff = np.abs(ours[valid] - theirs[valid])
        diff_signal = np.abs(ours[signal] - theirs[signal])
        scale = 100.0 if percent else 1.0
        rows.append(
            {
                "file": name,
                "software": software,
                "dark": dark is not None,
                "unit": unit or "fraction",
                "points": int(valid.size),
                "compared": int(valid.sum()),
                "max_abs": float(diff.max()),
                "median_abs": float(np.median(diff)),
                "max_abs_fraction": float(diff.max() / scale),
                "within_1e-6_of_full_scale": float(np.mean(diff <= 1e-6 * scale)),
                "compared_with_signal": int(signal.sum()),
                "max_abs_fraction_with_signal": float(diff_signal.max() / scale),
            }
        )
    for row in rows:
        print(
            f"{row['file']:<55} {row['software']:<26} max={row['max_abs_fraction']:.2e} "
            f"median={row['median_abs'] / (100.0 if '%' in row['unit'] else 1.0):.2e} "
            f"with-signal max={row['max_abs_fraction_with_signal']:.2e}"
        )
    REPORT.write_text(json.dumps({"cases": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
