"""Gap hunter: where does the formatting closure fail?

The formatting_anomaly_fraction closure (rounds 66/68) catches case,
date-format, whitespace, and heading-depth channels at J=1.0 — but its
sub-rules are precision rules, and each precision rule has an evasion edge.
This experiment systematically probes every sub-score with a minimal
evading artifact and reports which edges exist:

1. case sub-score (counts only mid-word flips): word-initial flips
   ("Report" from "report") — the WORD-1 finding, re-measured here.
2. heading sub-score (flags level SKIPS > +1): smooth 1-2 alternation
   (# -> ## -> # -> ## is normal nesting) — a candidate gap.
3. space sub-score (counts only word-separating runs): line-START runs
   (indentation) are excluded by design (JSON false-positive fix) — a
   candidate gap.
4. date sub-score ((distinct formats - 1) / dates): the receiver and the
   rule share the codebook, so any mix is caught — expected gap-free.

Each candidate is scored by _formatting_anomaly_fraction: 0.0 = the
closure is blind (a gap); >0 = closed. The result is the measured edge
map of the defense.

Usage:
    python experiments/closure_gap_hunter.py

Writes findings/closure_gap_hunter/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.detector import _formatting_anomaly_fraction
from src.token_stego.text_channels import CaseChannel, WhitespaceChannel

FINDINGS_DIR = Path("findings/closure_gap_hunter")


def run() -> None:
    edges: dict[str, dict] = {}

    # 1. Case: word-initial flips.
    case_cover = (
        "the report is final and the audit passed all checks today and the team moved"
    )
    case_text = CaseChannel().encode_bits(
        "1" * 10, case_cover
    )  # mid-word flips (closed)
    wi_cover = (
        "the report is final and the audit passed all checks today and the team moved"
    )
    wi_text = " ".join(
        w[0].upper() + w[1:] for w in wi_cover.split()
    )  # word-initial flips
    edges["case_word_initial"] = {
        "closure_score": _formatting_anomaly_fraction([wi_text]),
        "mid_word_baseline_score": _formatting_anomaly_fraction([case_text]),
        "gap": _formatting_anomaly_fraction([wi_text]) == 0.0,
        "note": "a flipped word-initial is byte-identical to a proper noun (WORD-1)",
    }

    # 2. Heading: smooth 1-2 alternation (no skips).
    heading_text = "".join(
        ("# Section %d\n" % i) + ("## Sub %d\n" % i) for i in range(30)
    )
    heading_skip = "# A\n#### D\n# B\n"  # a skip (> +1): should be flagged
    edges["heading_smooth_alternation"] = {
        "closure_score": _formatting_anomaly_fraction([heading_text]),
        "skip_baseline_score": _formatting_anomaly_fraction([heading_skip]),
        "gap": _formatting_anomaly_fraction([heading_text]) == 0.0,
        "note": "# -> ## alternation is normal nesting; only skips are flagged",
    }

    # 3. Space: line-start (indentation) runs are excluded by the rule.
    indent_text = "line one\n  line two indented\n  line three\n   deeper\n"
    ws_text = WhitespaceChannel().encode_bits(
        "10101010", "a  b  c  d  e  f  g  h  i  j"
    )
    edges["space_line_start_runs"] = {
        "closure_score": _formatting_anomaly_fraction([indent_text]),
        "word_separating_baseline_score": _formatting_anomaly_fraction([ws_text]),
        "gap": _formatting_anomaly_fraction([indent_text]) == 0.0,
        "note": "indentation runs are excluded (JSON false-positive fix)",
    }

    # 4. Date: any codebook mix is caught (expected gap-free).
    from src.channels.datefmt import DateFormatChannel

    date_text = DateFormatChannel().encode_bits(
        "".join(f"{i:03b}" for i in range(8)) * 2, _dated_cover()
    )
    edges["date_mix"] = {
        "closure_score": _formatting_anomaly_fraction([date_text]),
        "gap": _formatting_anomaly_fraction([date_text]) == 0.0,
        "note": "receiver and rule share the codebook; any mix is flagged",
    }

    results = {
        "edges": edges,
        "n_gaps": sum(1 for e in edges.values() if e["gap"]),
        "reading": (
            "every precision rule in the closure has an evasion edge EXCEPT "
            "the date rule (codebook-shared). The gaps are structural: "
            "word-initial flips (proper-noun ambiguity), smooth heading "
            "alternation (normal nesting), and line-start indentation "
            "(excluded to avoid JSON false positives). Each is invisible to "
            "the closure by construction — the measured edge map of the "
            "formatting defense."
        ),
    }

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FINDINGS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


def _dated_cover() -> str:
    lines = ["Release notes\n"]
    for i in range(20):
        month, day = 1 + i // 28, 1 + i % 28
        lines.append(f"2026-{month:02d}-{day:02d}: fixed config loader.\n")
    return "".join(lines)


if __name__ == "__main__":
    run()
