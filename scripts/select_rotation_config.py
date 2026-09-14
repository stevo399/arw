"""Apply the pre-registered selection rule to a rotation sweep (plan 2026-09-14, Task 7).

Among swept configurations, choose the most tornado event hits at rank 1 whose
null-storm false-alarm fraction at rank 1 does not exceed the baseline's.
Ties: lower null-storm false-alarm fraction, then more NMD matches, then grid order.
Usage: select_rotation_config.py SWEEP.json [BASELINE_OUT.json]
"""
import json
import sys


def null_fraction(entry, rank="1"):
    s = entry["summary"][rank]
    return s["null_false_alarms"] / max(s["null_probes"], 1)


def main(path, baseline_out=None):
    entries = json.load(open(path, encoding="utf-8"))
    baseline, candidates = entries[0], entries[1:]
    if baseline_out:
        json.dump(baseline, open(baseline_out, "w", encoding="utf-8"), indent=2)
    limit = null_fraction(baseline)
    header = f"{'#':>2} {'shear':>5} {'side':>4} {'diam':>5} {'fold':>5} | {'hits r1':>7} {'r2':>3} {'r3':>3} | {'null r1':>12} | {'NMD':>3} | {'clear r1':>8} | {'speak':>11} | {'sec':>6}"
    print(header)
    rows = [("base", baseline)] + list(enumerate(candidates))
    for index, entry in rows:
        c, s = entry["config"], entry["summary"]
        speak = "".join(str(r) for r in ("1", "2", "3") if s[r]["speakable"]) or "-"
        print(f"{index!s:>2} {c['min_shear_ms']:>5} {c['min_side_ms']:>4} {str(c['max_diameter_km']):>5} {str(c['fold_rejection']):>5} | "
              f"{s['1']['tornado_hits']:>3}/{s['1']['tornado_events']:<3} {s['2']['tornado_hits']:>3} {s['3']['tornado_hits']:>3} | "
              f"{s['1']['null_false_alarms']:>4}/{s['1']['null_probes']:<5} {null_fraction(entry):.3f} | {s['1']['nmd_matched']:>3} | "
              f"{s['1']['clear_air_false_alarms']:>8} | {speak:>11} | {entry['analysis_seconds']:>6}")
    eligible = [(i, e) for i, e in enumerate(candidates) if null_fraction(e) <= limit]
    if not eligible:
        print("SELECTED: none within the baseline null false-alarm fraction; use the physics corrections only")
        return None
    index, chosen = max(eligible, key=lambda item: (item[1]["summary"]["1"]["tornado_hits"], -null_fraction(item[1]),
                                                     item[1]["summary"]["1"]["nmd_matched"], -item[0]))
    print("SELECTED:", index, json.dumps(chosen["config"]))
    return chosen


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
