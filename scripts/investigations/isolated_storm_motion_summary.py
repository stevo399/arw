"""Summarize isolated_storm_motion_eval.py output: which acceptance rule for storms with fewer than
three judging neighbours predicts the next scan better than assuming no motion?

Each rule accepts or rejects a raw match; a rejected storm is scored as not
moving (what reporting would fall back to without other information).  Rules
are compared on identical targets: isolated storms with a raw match within the
150 km/h limit whose next position was observed and was not a merge, split or
reacquisition.
"""
import json
import math
import statistics
import sys

import numpy as np

STRUCTURAL = {"merge_survivor", "split_child", "reacquired"}
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]


def agrees(row, tolerance):
    previous = row["previous_raw"]
    if previous is None:
        return False
    return math.hypot(row["raw"][0] - previous[0], row["raw"][1] - previous[1]) <= tolerance


RULES = {
    "accept_all": lambda r: True,
    "corr>=0.5": lambda r: r["raw"][2] >= 0.5,
    "corr>=0.6": lambda r: r["raw"][2] >= 0.6,
    "corr>=0.7": lambda r: r["raw"][2] >= 0.7,
    "corr>=0.8": lambda r: r["raw"][2] >= 0.8,
    "speed<=100": lambda r: r["raw_speed_kmh"] <= 100.0,
    "agrees_with_previous_25": lambda r: agrees(r, 25.0),
    "agrees_with_previous_15": lambda r: agrees(r, 15.0),
    "agrees_25_and_corr>=0.6": lambda r: agrees(r, 25.0) and r["raw"][2] >= 0.6,
    "reject_all": lambda r: False,
}


def score(subset, rule):
    errors, ious, accepted = [], [], 0
    for r in subset:
        use = rule(r)
        accepted += use
        chosen = r["results"]["raw" if use else "stationary"]
        errors.append(chosen["centre_error_km"])
        ious.append(chosen["iou"])
    return {
        "accepted_share": round(accepted / len(subset), 3),
        "median_error_km": round(statistics.median(errors), 3),
        "p90_error_km": round(float(np.percentile(errors, 90)), 3),
        "mean_iou": round(statistics.fmean(ious), 4),
    }


def eligible(r):
    return (
        r["raw"][3] is False
        and r["raw_speed_kmh"] <= 150.0
        and r.get("target_event") not in STRUCTURAL
    )


summary = {"targets": len(rows)}
for label, condition in (
    ("isolated (fewer than 3 neighbours)", lambda r: r["neighbours"] < 3),
    ("judged by neighbours (3 or more)", lambda r: r["neighbours"] >= 3),
):
    subset = [r for r in rows if eligible(r) and condition(r)]
    summary[label] = {"n": len(subset), "rules": {name: score(subset, rule) for name, rule in RULES.items()}}
    with_previous = [r for r in subset if r["previous_raw"] is not None]
    if with_previous:
        summary[label]["with_previous_raw"] = {
            "n": len(with_previous),
            "rules": {name: score(with_previous, rule) for name, rule in RULES.items()},
        }

with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=1))
