#!/usr/bin/env python3
"""TUNE-1 exact statistic, flags and deterministic selection (tuning protocol v2, limber).

Inputs are exact `Fraction` values (Criterion `estimates.json` `median` point estimates
and 95% bounds, in nanoseconds). Every ratio is reduced to an integer pair and compared
by cross multiplication; no float is ever formed. `tuning_result(...)` renders the
canonical `tuning-result.json` object whose exact-byte SHA-256 is the group-local
`tuning_id`.

limber difference from the Zinc+ module: the schedule (canonical) candidate order is the
pinned k order `[10, 7, 12, 9, 13, 8, 11]` while the simplicity order is ascending k, so the
decision takes an explicit `simplicity_order` (`s0` = its first member, `p` = its first
member inside the band); the result records both lists. Without a `simplicity_order` the
candidate order is the simplicity order, exactly as in Zinc's module.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

try:
    from scripts import poseidon_common as pc
except ImportError:  # executed as a plain file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import poseidon_common as pc  # noqa: E402

TUNING_RESULT_SCHEMA = "limber/poseidon-tuning-result/v2"
SELECTION_LABELS = ("only_admissible", "simplest_unique_best", "simplicity_tie_band",
                    "performance", "simplicity_insufficient_wins")
BAND_NUMERATOR = 103
BAND_DENOMINATOR = 100
WINS_THRESHOLD = 7


def block_orders(candidates) -> list:
    """The `2N` counterbalanced block orders for an ordered candidate list."""
    cands = list(candidates)
    n = len(cands)
    if n == 0:
        raise ValueError("no candidates")
    orders = []
    for b in range(n):
        orders.append(cands[b:] + cands[:b])
    rev = list(reversed(cands))
    for b in range(n):
        orders.append(rev[b:] + rev[:b])
    return orders


def schedule(candidates, instances) -> list:
    """Flat execution schedule: `[(block, ordinal, candidate, instance)]` (2N blocks of
    9N processes = 18N^2 children for N candidates and nine instances)."""
    out = []
    for b, order in enumerate(block_orders(candidates)):
        ordinal = 0
        for inst in instances:
            for cand in order:
                out.append((b, ordinal, cand, inst))
                ordinal += 1
    return out


def child_dir_name(ordinal: int, candidate, instance: int) -> str:
    """`<ordinal>-k<candidate>-inst<instance>` inside `children/block-<b>/`."""
    return "%02d-k%s-inst%d" % (ordinal, candidate, instance)


def _simplicity(candidates, simplicity_order) -> list:
    cands = list(candidates)
    order = list(simplicity_order) if simplicity_order is not None else list(cands)
    if sorted(map(str, order)) != sorted(map(str, cands)) or len(order) != len(cands):
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "simplicity order %r is not a permutation of the candidates %r" %
                             (order, cands))
    return order


def decide(candidates, instances, processes: dict, simplicity_order=None) -> dict:
    """Apply the TUNE-1 rule.

    `candidates`: canonical (schedule) order; `simplicity_order`: the simplicity order
    (default: the candidate order). `processes[(cand, inst, block)]` is a dict with `point`,
    `lower`, `upper` (positive `Fraction`). Every (cand, inst) needs all `2N` blocks. Returns
    exact values (`Fraction`) plus the labelled decision.
    """
    cands = list(candidates)
    simple = _simplicity(cands, simplicity_order)
    insts = list(instances)
    n = len(cands)
    nblocks = 2 * n
    s0 = simple[0]
    pair_median = {}
    envelope = {}
    for c in cands:
        for i in insts:
            points, lowers, uppers = [], [], []
            for b in range(nblocks):
                rec = processes.get((c, i, b))
                if rec is None:
                    raise pc.RunnerError("aggregate", "TuningResultInvalid",
                                         "missing process estimate for %s inst%d block%d" %
                                         (c, i, b))
                points.append(pc.positive_fraction(rec["point"], "point"))
                lowers.append(pc.positive_fraction(rec["lower"], "lower"))
                uppers.append(pc.positive_fraction(rec["upper"], "upper"))
            pair_median[(c, i)] = pc.exact_median(points)
            envelope[(c, i)] = (min(lowers), max(uppers))
    ratio = {(c, i): pair_median[(c, i)] / pair_median[(s0, i)] for c in cands for i in insts}
    score = {c: pc.exact_median([ratio[(c, i)] for i in insts]) for c in cands}
    min_score = min(score.values())
    band = [c for c in simple if BAND_DENOMINATOR * score[c] <= BAND_NUMERATOR * min_score]
    wins = {c: sum(1 for i in insts if ratio[(c, i)] < 1) for c in cands}
    parity = {}
    for c in cands:
        count = 0
        for i in insts:
            lo_c, up_c = envelope[(c, i)]
            lo_s, up_s = envelope[(s0, i)]
            if lo_c / up_s <= 1 <= up_c / lo_s:
                count += 1
        parity[c] = count
    p = band[0]
    if n == 1:
        selected, basis = s0, "only_admissible"
    elif band == [s0]:
        selected, basis = s0, "simplest_unique_best"
    elif s0 in band and len(band) > 1:
        selected, basis = s0, "simplicity_tie_band"
    elif s0 not in band and wins[p] >= WINS_THRESHOLD:
        selected, basis = p, "performance"
    else:
        selected, basis = s0, "simplicity_insufficient_wins"
    return {
        "candidates": cands, "simplicity_order": simple, "instances": insts, "simplest": s0,
        "pair_median": pair_median, "envelope": envelope, "ratio": ratio, "score": score,
        "min_score": min_score, "band": band, "wins_vs_simplest": wins,
        "ci_parity_count": parity, "tie_band_non_singleton": len(band) > 1,
        "selected": selected, "selection_basis": basis,
    }


def render_decision(d: dict) -> dict:
    """Canonical-JSON form of `decide(...)` output (decimal strings and reduced pairs).
    Per-candidate maps are keyed by the decimal rendering of k."""
    cands, insts = d["candidates"], d["instances"]
    per_pair = []
    for c in cands:
        for i in insts:
            lo, up = d["envelope"][(c, i)]
            per_pair.append({
                "candidate": c, "instance": i,
                "median_ns": pc.fraction_to_decimal(d["pair_median"][(c, i)]),
                "ratio_to_simplest": pc.ratio_pair(d["ratio"][(c, i)]),
                "envelope_lower_ns": pc.fraction_to_decimal(lo),
                "envelope_upper_ns": pc.fraction_to_decimal(up),
            })
    return {
        "candidates": cands, "simplicity_order": d["simplicity_order"], "instances": insts,
        "simplest": d["simplest"],
        "per_candidate_instance": per_pair,
        "scores": {str(c): pc.ratio_pair(d["score"][c]) for c in cands},
        "min_score": pc.ratio_pair(d["min_score"]),
        "band": d["band"], "band_rule": "100*score <= 103*min_score",
        "wins_vs_simplest": {str(c): d["wins_vs_simplest"][c] for c in cands},
        "ci_parity_count": {str(c): d["ci_parity_count"][c] for c in cands},
        "tie_band_non_singleton": d["tie_band_non_singleton"],
        "wins_threshold": WINS_THRESHOLD,
        "selected": d["selected"], "selection_basis": d["selection_basis"],
    }


def process_records(processes: dict) -> list:
    out = []
    for (c, i, b), rec in sorted(processes.items(), key=lambda kv: (kv[0][2], kv[0][1],
                                                                     str(kv[0][0]))):
        out.append({"candidate": c, "instance": i, "block": b,
                    "median_ns": pc.fraction_to_decimal(rec["point"]),
                    "lower_ns": pc.fraction_to_decimal(rec["lower"]),
                    "upper_ns": pc.fraction_to_decimal(rec["upper"])})
    return out


def tuning_result(group: dict, ids: dict, candidates, instances, processes: dict,
                  provenance: dict, simplicity_order=None) -> dict:
    """The canonical `tuning-result.json` object (its SHA-256 is `tuning_id`)."""
    decision = decide(candidates, instances, processes, simplicity_order)
    orders = block_orders(candidates)
    return {
        "schema": TUNING_RESULT_SCHEMA,
        "protocol": "TUNE-1",
        "ids": ids,
        "group": group,
        "candidate_parameter": "k",
        "hashes_per_field": pc.CANONICAL_HASHES,
        "candidates": list(candidates),
        "simplicity_order": decision["simplicity_order"],
        "instances": list(instances),
        "blocks": [{"block": b, "order": order} for b, order in enumerate(orders)],
        "statistic": {"group": "prove_e2e", "estimate": "median.point_estimate",
                      "interval": "median.confidence_interval", "unit": "ns"},
        "process_estimates": process_records(processes),
        "aggregate": render_decision(decision),
        "decision": {"selected": decision["selected"],
                     "selection_basis": decision["selection_basis"]},
        "provenance": provenance,
    }


def recheck_tuning_result(obj: dict) -> None:
    """Re-execute the decision recorded in a tuning result and require equality."""
    try:
        cands = obj["candidates"]
        insts = obj["instances"]
        simple = obj.get("simplicity_order", cands)
        processes = {}
        for rec in obj["process_estimates"]:
            processes[(rec["candidate"], rec["instance"], rec["block"])] = {
                "point": pc.fraction_from_decimal(rec["median_ns"]),
                "lower": pc.fraction_from_decimal(rec["lower_ns"]),
                "upper": pc.fraction_from_decimal(rec["upper_ns"]),
            }
        expected_blocks = [{"block": b, "order": o} for b, o in enumerate(block_orders(cands))]
        if obj["blocks"] != expected_blocks:
            raise ValueError("block schedule does not match the TUNE-1 rule")
        if len(processes) != len(cands) * len(insts) * 2 * len(cands):
            raise ValueError("process count mismatch")
        decision = decide(cands, insts, processes, simple)
        if render_decision(decision) != obj["aggregate"]:
            raise ValueError("aggregate does not reproduce")
        if obj["decision"] != {"selected": decision["selected"],
                               "selection_basis": decision["selection_basis"]}:
            raise ValueError("decision does not reproduce")
    except (KeyError, TypeError, ValueError) as exc:
        raise pc.RunnerError("aggregate", "TuningResultInvalid", str(exc)) from None
