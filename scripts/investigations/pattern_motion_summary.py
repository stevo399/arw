"""Summarize pattern_motion_eval.py output.

For each group (earlier positions 1, 2, 3, 4+; clean or structural target), and
for each candidate, reports centre error and outline IoU -- both over all targets
where the candidate exists, and head-to-head against stationary and
clean_steps_3 on exactly the same targets, so availability does not bias the
comparison.  Also splits pattern results by the match's correlation strength.
"""
import json
import statistics
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
CANDIDATES = sorted({name for row in rows for name in row["results"]})


def group_names(row):
    prior = row["prior_positions"]
    prior = "1" if prior == 1 else "2" if prior == 2 else "3" if prior == 3 else "4+"
    target = "clean" if row["clean_target"] else "structural"
    return ["all", f"prior_{prior}", f"prior_{prior}_{target}", f"all_{target}"]


def stats(values):
    return {
        "n": len(values),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
    }


summary = {"targets": len(rows), "groups": {}, "head_to_head": {}, "pattern_by_ncc": {}}
by_group = defaultdict(lambda: defaultdict(lambda: {"error": [], "iou": []}))
for row in rows:
    for group in group_names(row):
        for name, result in row["results"].items():
            by_group[group][name]["error"].append(result["centre_error_km"])
            by_group[group][name]["iou"].append(result["iou"])
for group, candidates in sorted(by_group.items()):
    summary["groups"][group] = {
        name: {"centre_error_km": stats(v["error"]), "iou": stats(v["iou"])}
        for name, v in candidates.items() if v["error"]
    }

# Head to head: targets where both candidates exist.
for baseline in ("stationary", "clean_steps_3", "pattern_last"):
    for challenger in CANDIDATES:
        if challenger == baseline:
            continue
        for group_filter in ("all", "all_clean", "prior_1", "prior_2", "prior_3", "prior_4+"):
            pairs = [
                (row["results"][baseline], row["results"][challenger])
                for row in rows
                if baseline in row["results"] and challenger in row["results"] and group_filter in group_names(row)
            ]
            if not pairs:
                continue
            error_gain = [b["centre_error_km"] - c["centre_error_km"] for b, c in pairs]
            iou_gain = [c["iou"] - b["iou"] for b, c in pairs]
            summary["head_to_head"][f"{challenger} vs {baseline} [{group_filter}]"] = {
                "n": len(pairs),
                "median_error_baseline": round(statistics.median(b["centre_error_km"] for b, _ in pairs), 3),
                "median_error_challenger": round(statistics.median(c["centre_error_km"] for _, c in pairs), 3),
                "p90_error_baseline": round(float(np.percentile([b["centre_error_km"] for b, _ in pairs], 90)), 3),
                "p90_error_challenger": round(float(np.percentile([c["centre_error_km"] for _, c in pairs], 90)), 3),
                "mean_iou_baseline": round(statistics.fmean(b["iou"] for b, _ in pairs), 4),
                "mean_iou_challenger": round(statistics.fmean(c["iou"] for _, c in pairs), 4),
                "challenger_better_iou_share": round(sum(g > 0 for g in iou_gain) / len(pairs), 3),
                "challenger_worse_iou_share": round(sum(g < 0 for g in iou_gain) / len(pairs), 3),
                "median_error_gain_km": round(statistics.median(error_gain), 3),
            }

for low, high in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
    subset = [row for row in rows if row["last_pattern_ncc"] is not None and low <= row["last_pattern_ncc"] < high and "pattern_last" in row["results"]]
    if subset:
        summary["pattern_by_ncc"][f"{low}-{high}"] = {
            "n": len(subset),
            "pattern_last_error": stats([r["results"]["pattern_last"]["centre_error_km"] for r in subset]),
            "stationary_error": stats([r["results"]["stationary"]["centre_error_km"] for r in subset]),
            "pattern_last_iou_mean": round(statistics.fmean(r["results"]["pattern_last"]["iou"] for r in subset), 4),
            "stationary_iou_mean": round(statistics.fmean(r["results"]["stationary"]["iou"] for r in subset), 4),
        }

with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps({"targets": summary["targets"], "head_to_head": summary["head_to_head"], "pattern_by_ncc": summary["pattern_by_ncc"]}, indent=1))
