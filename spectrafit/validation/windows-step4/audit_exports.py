"""Independent host-only MAT/TIFF read-back, beyond the no-Python kit checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.io import loadmat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "export-audit.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    fixtures = root / ".windows-baseline/step4-fixtures"
    definitions = json.loads((fixtures / "cases.json").read_text(encoding="utf-8"))
    counts = {"mat_arrays": 0, "tiff_arrays": 0}
    time_tags, issues = [], []
    for case in definitions["cases"]:
        folder = args.results / "cases" / case["name"]
        mat = loadmat(folder / "results.mat")
        with np.load(fixtures / "expected" / (case["name"] + ".npz"), allow_pickle=False) as expected:
            for name in expected.files:
                if name == "wavelengths":
                    continue
                values = expected[name]
                mapped = values[0] if values.shape[0] == 1 else np.moveaxis(values, 0, -1)
                np.testing.assert_array_equal(mat[name], mapped)
                counts["mat_arrays"] += 1
                path = folder / "tiff" / (name + ".tif")
                np.testing.assert_array_equal(tifffile.imread(path), values[0] if values.shape[0] == 1 else values)
                counts["tiff_arrays"] += 1
                if values.shape[0] > 1:
                    with tifffile.TiffFile(path) as file:
                        tag = {"case": case["name"], "map": name, "frames": values.shape[0],
                               "pages": len(file.pages), "photometric": str(file.pages[0].photometric),
                               "axes": file.series[0].axes}
                        time_tags.append(tag)
                        if len(file.pages) != values.shape[0] or int(file.pages[0].photometric) != 1:
                            issues.append(tag)
    report = {"numeric_roundtrip": "passed", "counts": counts, "time_series_tags": time_tags,
              "time_series_semantics_issues": issues}
    target = args.report
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"numeric_roundtrip": "passed", **counts, "time_series_semantics_issues": len(issues),
                      "first_time_series_tag": time_tags[0]}, indent=2))
    raise SystemExit(1 if issues else 0)
