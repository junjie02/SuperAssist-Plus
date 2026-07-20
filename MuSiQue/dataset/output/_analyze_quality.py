#!/usr/bin/env python3
"""
Data-quality analysis of generated MuSiQue trajectories.
Covers all 12 dimensions plus a few cross-tabulations.
"""

import json, collections, sys, re
from pathlib import Path

INPUT = Path(r"f:\cognifold dataset\CogniFold\dataset\output\musique_trajectories.jsonl")
# Pad width for aligned output
W = 55

# ──────────────────────────────────────────────
# Load everything first
# ──────────────────────────────────────────────
records = []
with open(INPUT, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

N = len(records)
print(f"{'Total trajectories':<{W}} {N}")

# ──────────────────────────────────────────────
# Helper: get parsed_plan.operations safely
# ──────────────────────────────────────────────
def get_plan(rec):
    """Return parsed_plan dict (or empty dict if None/missing)."""
    plan = rec.get("output", {}).get("parsed_plan")
    if plan is None:
        return {}
    return plan
def get_ops_list(rec):
    plan = get_plan(rec)
    return plan.get("operations", []) or []

# ══════════════════════════════════════════════
# 1. Parse errors
# ══════════════════════════════════════════════
parse_errors = [r for r in records if r.get("output", {}).get("parse_error") is not None]
print(f"\n{'1. Parse errors':-<{W}}")
print(f"{'  Count with parse_error != null':<{W}} {len(parse_errors)} ({len(parse_errors)/N*100:.1f}%)")
# Show unique error messages
err_msgs = collections.Counter()
for r in parse_errors:
    err_msgs[r["output"]["parse_error"]] += 1
print(f"{'  Unique error messages':<{W}} {len(err_msgs)}")
for msg, cnt in err_msgs.most_common(5):
    print(f"    [{cnt:4d}] {msg[:120]}")

# ══════════════════════════════════════════════
# 2. Empty response
# ══════════════════════════════════════════════
empty_resp = [r for r in records if not (r.get("output", {}).get("raw_response") or "").strip()]
print(f"\n{'2. Empty/whitespace raw_response':-<{W}}")
print(f"{'  Count':<{W}} {len(empty_resp)} ({len(empty_resp)/N*100:.1f}%)")

# ══════════════════════════════════════════════
# 3. Operations count distribution
# ══════════════════════════════════════════════
ops_bins = collections.Counter()
zero_ops = 0
for r in records:
    n_ops = len(get_ops_list(r))
    if n_ops == 0:
        zero_ops += 1
    if n_ops <= 5:
        ops_bins[n_ops] += 1
    elif n_ops <= 10:
        ops_bins["6-10"] += 1
    elif n_ops <= 20:
        ops_bins["11-20"] += 1
    else:
        ops_bins["21+"] += 1

print(f"\n{'3. Operations count (parsed_plan.operations)':-<{W}}")
print(f"{'  Trajectories with 0 ops':<{W}} {zero_ops} ({zero_ops/N*100:.1f}%)")
for k in sorted(ops_bins.keys(), key=lambda x: (isinstance(x, str), x)):
    print(f"    ops = {str(k):>6}: {ops_bins[k]:5d} ({ops_bins[k]/N*100:5.1f}%)")

# ══════════════════════════════════════════════
# 4. Operation types
# ══════════════════════════════════════════════
op_type_counter = collections.Counter()
for r in records:
    for op in get_ops_list(r):
        op_type_counter[op.get("op", "MISSING_OP")] += 1

print(f"\n{'4. Operation type frequencies':-<{W}}")
for op_type, cnt in op_type_counter.most_common():
    print(f"    {op_type:<25} {cnt:6d}")

# ══════════════════════════════════════════════
# 5. Symbolic actions
# ══════════════════════════════════════════════
non_empty_sym = 0
sym_action_counts = []
for r in records:
    plan = get_plan(r)
    sa = plan.get("symbolic_actions") or []
    # Also check top-level
    if not sa:
        sa_raw = r.get("output", {}).get("raw_response", "")
        if sa_raw:
            try:
                parsed = json.loads(sa_raw)
                sa = parsed.get("symbolic_actions") or []
            except Exception:
                pass
    sym_action_counts.append(len(sa))
    if sa:
        non_empty_sym += 1

print(f"\n{'5. Symbolic actions':-<{W}}")
print(f"{'  Non-empty symbolic_actions':<{W}} {non_empty_sym} ({non_empty_sym/N*100:.1f}%)")
if sym_action_counts:
    sa_dist = collections.Counter(sym_action_counts)
    print(f"{'  Count distribution (sa_count -> n_trajs):':<{W}}")
    for k in sorted(sa_dist.keys()):
        print(f"    {k:>4} action(s): {sa_dist[k]:5d}")

# ══════════════════════════════════════════════
# 6. Reasoning quality
# ══════════════════════════════════════════════
no_reasoning = 0
short_reasoning = 0  # present but <= 20 chars
good_reasoning = 0
reasoning_lengths = []
for r in records:
    plan = get_plan(r)
    reasoning = (plan.get("reasoning") or "").strip()
    if not reasoning:
        no_reasoning += 1
    elif len(reasoning) <= 20:
        short_reasoning += 1
    else:
        good_reasoning += 1
    reasoning_lengths.append(len(reasoning))

print(f"\n{'6. Reasoning quality (parsed_plan.reasoning)':-<{W}}")
print(f"{'  Missing/empty':<{W}} {no_reasoning} ({no_reasoning/N*100:.1f}%)")
print(f"{'  Present but <= 20 chars':<{W}} {short_reasoning} ({short_reasoning/N*100:.1f}%)")
print(f"{'  Good (> 20 chars)':<{W}} {good_reasoning} ({good_reasoning/N*100:.1f}%)")
if reasoning_lengths:
    print(f"{'  Mean length':<{W}} {sum(reasoning_lengths)/len(reasoning_lengths):.1f}")
    print(f"{'  Median length':<{W}} {sorted(reasoning_lengths)[len(reasoning_lengths)//2]}")
    print(f"{'  P5 / P95 length':<{W}} {sorted(reasoning_lengths)[max(0,len(reasoning_lengths)//20)]} / {sorted(reasoning_lengths)[len(reasoning_lengths)*95//100]}")

# ══════════════════════════════════════════════
# 7. Grounding quality (non-event ADD_NODEs)
# ══════════════════════════════════════════════
total_non_event_adds = 0
grounded_ok = 0
grounded_missing = 0
grounded_empty_list = 0
for r in records:
    for op in get_ops_list(r):
        if op.get("op") != "ADD_NODE":
            continue
        if op.get("node_type") == "event":
            continue
        total_non_event_adds += 1
        gi = op.get("grounded_in")
        if gi is None or (isinstance(gi, list) and len(gi) == 0):
            grounded_empty_list += 1
        else:
            grounded_ok += 1

print(f"\n{'7. Grounding quality (non-event ADD_NODEs)':-<{W}}")
print(f"{'  Total non-event ADD_NODEs':<{W}} {total_non_event_adds}")
if total_non_event_adds > 0:
    print(f"{'  grounded_in present & non-empty':<{W}} {grounded_ok} ({grounded_ok/total_non_event_adds*100:.1f}%)")
    print(f"{'  grounded_in missing or empty list':<{W}} {grounded_empty_list} ({grounded_empty_list/total_non_event_adds*100:.1f}%)")

def _extract_node_id(op):
    """Extract the node identifier from an ADD_NODE operation."""
    data = op.get("data") or {}
    nid = (data.get("event_id") or data.get("concept_id") or
           data.get("intent_id") or data.get("id") or data.get("node_id") or
           op.get("event_id") or op.get("node_id"))
    return nid

# ══════════════════════════════════════════════
# 8. Edge quality — CONCEPT/INTENT node connectivity
# ══════════════════════════════════════════════
# For each trajectory, check: for every non-event ADD_NODE, is there at least
# one ADD_EDGE that references that node's id as source or target?
orphan_count = 0
total_non_event = 0
trajs_with_orphans = 0
for r in records:
    ops = get_ops_list(r)
    # Collect all ADD_EDGE sources and targets
    edge_sources = set()
    edge_targets = set()
    non_event_ids = []
    event_ids = []
    for op in ops:
        if op.get("op") == "ADD_EDGE":
            edge_sources.add(op.get("source_id"))
            edge_targets.add(op.get("target_id"))
        elif op.get("op") == "ADD_NODE":
            nid = _extract_node_id(op)
            if not nid:
                continue
            if op.get("node_type") == "event":
                event_ids.append(nid)
            else:
                non_event_ids.append(nid)
    # Also collect UPDATE_NODE references (they connect too, but edge is the main requirement)
    traj_orphans = 0
    for nid in non_event_ids:
        total_non_event += 1
        if nid not in edge_sources and nid not in edge_targets:
            orphan_count += 1
            traj_orphans += 1
    if traj_orphans > 0:
        trajs_with_orphans += 1

print(f"\n{'8. Edge quality (non-event ADD_NODE connectivity)':-<{W}}")
print(f"{'  Total non-event nodes':<{W}} {total_non_event}")
if total_non_event > 0:
    print(f"{'  Orphan nodes (no edge referencing them)':<{W}} {orphan_count} ({orphan_count/total_non_event*100:.1f}%)")
print(f"{'  Trajectories with >=1 orphan':<{W}} {trajs_with_orphans} ({trajs_with_orphans/N*100:.1f}%)")

# ══════════════════════════════════════════════
# 9. Output JSON validity
# ══════════════════════════════════════════════
json_valid = 0
json_invalid = 0
json_empty = 0
json_invalid_samples = []
for r in records:
    rr = (r.get("output", {}).get("raw_response") or "").strip()
    if not rr:
        json_empty += 1
        continue
    try:
        json.loads(rr)
        json_valid += 1
    except json.JSONDecodeError as e:
        json_invalid += 1
        if len(json_invalid_samples) < 5:
            json_invalid_samples.append((r["trajectory_id"], str(e), rr[:200]))

print(f"\n{'9. Output JSON validity':-<{W}}")
print(f"{'  Valid JSON':<{W}} {json_valid} ({json_valid/N*100:.1f}%)")
print(f"{'  Invalid JSON':<{W}} {json_invalid} ({json_invalid/N*100:.1f}%)")
print(f"{'  Empty (no raw_response)':<{W}} {json_empty} ({json_empty/N*100:.1f}%)")
if json_invalid_samples:
    print(f"{'  Sample invalid (first 5):':<{W}}")
    for tid, err, snippet in json_invalid_samples:
        print(f"    [{tid}] {err}")

# ══════════════════════════════════════════════
# 10. Input size distribution
# ══════════════════════════════════════════════
input_sizes = []
for r in records:
    sp = (r.get("input", {}).get("system_prompt") or "")
    up = (r.get("input", {}).get("user_prompt") or "")
    input_sizes.append(len(sp) + len(up))

print(f"\n{'10. Input size (system_prompt + user_prompt chars)':-<{W}}")
if input_sizes:
    input_sizes.sort()
    print(f"{'  Mean':<{W}} {sum(input_sizes)/len(input_sizes):.0f}")
    print(f"{'  Median':<{W}} {input_sizes[len(input_sizes)//2]}")
    print(f"{'  Min':<{W}} {input_sizes[0]}")
    print(f"{'  Max':<{W}} {input_sizes[-1]}")
    ps = [10, 25, 50, 75, 90, 95, 99]
    for p in ps:
        idx = int(len(input_sizes) * p / 100)
        print(f"{'  P' + str(p):<{W}} {input_sizes[min(idx, len(input_sizes)-1)]}")

# ══════════════════════════════════════════════
# 11. Output size distribution
# ══════════════════════════════════════════════
output_sizes = []
for r in records:
    rr = (r.get("output", {}).get("raw_response") or "")
    output_sizes.append(len(rr))

print(f"\n{'11. Output size (raw_response chars)':-<{W}}")
if output_sizes:
    output_sizes.sort()
    print(f"{'  Mean':<{W}} {sum(output_sizes)/len(output_sizes):.0f}")
    print(f"{'  Median':<{W}} {output_sizes[len(output_sizes)//2]}")
    print(f"{'  Min':<{W}} {output_sizes[0]}")
    print(f"{'  Max':<{W}} {output_sizes[-1]}")
    ps = [10, 25, 50, 75, 90, 95, 99]
    for p in ps:
        idx = int(len(output_sizes) * p / 100)
        print(f"{'  P' + str(p):<{W}} {output_sizes[min(idx, len(output_sizes)-1)]}")

# ══════════════════════════════════════════════
# 12. Duplicate event_ids
# ══════════════════════════════════════════════
eid_counts = collections.Counter()
for r in records:
    eid = r.get("event_id")
    if eid:
        eid_counts[eid] += 1
dupes = {eid: cnt for eid, cnt in eid_counts.items() if cnt > 1}
print(f"\n{'12. Duplicate event_id detection':-<{W}}")
print(f"{'  Unique event_ids':<{W}} {len(eid_counts)}")
print(f"{'  Duplicated event_ids':<{W}} {len(dupes)}")
if dupes:
    for eid, cnt in sorted(dupes.items(), key=lambda x: -x[1])[:10]:
        print(f"    {eid}: {cnt} occurrences")

# ══════════════════════════════════════════════
# BONUS: Cross-tabulations
# ══════════════════════════════════════════════

# B1. Parse error vs empty response
pe_and_empty = [r for r in records if r.get("output",{}).get("parse_error") and not (r.get("output",{}).get("raw_response","").strip())]
pe_no_empty = [r for r in records if r.get("output",{}).get("parse_error") and (r.get("output",{}).get("raw_response","").strip())]
print(f"\n{'BONUS cross-tabs':-<{W}}")
print(f"  Parse error AND empty response: {len(pe_and_empty)}")
print(f"  Parse error BUT non-empty response: {len(pe_no_empty)}")

# B2. Missing event ADD_NODE
missing_event_node = 0
for r in records:
    has_event_add = any(
        op.get("op") == "ADD_NODE" and op.get("node_type") == "event"
        for op in get_ops_list(r)
    )
    if not has_event_add:
        missing_event_node += 1
print(f"  Trajectories missing mandatory event ADD_NODE: {missing_event_node} ({missing_event_node/N*100:.1f}%)")

# B3. Edge without ADD_EDGE op
edge_type_counter = collections.Counter()
for r in records:
    for op in get_ops_list(r):
        if op.get("op") == "ADD_EDGE":
            et = op.get("edge_type") or "MISSING"
            edge_type_counter[et] += 1
print(f"  ADD_EDGE edge_type distribution:")
for et, cnt in edge_type_counter.most_common():
    print(f"    {et:<25} {cnt:6d}")

# B4. Node type distribution within ADD_NODE
node_type_counter = collections.Counter()
for r in records:
    for op in get_ops_list(r):
        if op.get("op") == "ADD_NODE":
            node_type_counter[op.get("node_type", "MISSING_NODE_TYPE")] += 1
print(f"  ADD_NODE node_type distribution:")
for nt, cnt in node_type_counter.most_common():
    print(f"    {nt:<25} {cnt:6d}")

# B5. MERGE_NODES, REMOVE_NODE, REMOVE_EDGE counts
special_ops = collections.Counter()
for r in records:
    for op in get_ops_list(r):
        if op.get("op") in ("MERGE_NODES", "REMOVE_NODE", "REMOVE_EDGE"):
            special_ops[op.get("op")] += 1
if special_ops:
    print(f"  Rare operation counts:")
    for op, cnt in special_ops.most_common():
        print(f"    {op:<25} {cnt:6d}")
else:
    print(f"  MERGE_NODES/REMOVE_NODE/REMOVE_EDGE: 0 occurrences")

# B6. Trajectories with both concept AND intent
has_both = 0
has_concept_only = 0
has_intent_only = 0
for r in records:
    types = set()
    for op in get_ops_list(r):
        if op.get("op") == "ADD_NODE":
            types.add(op.get("node_type"))
    if "concept" in types and "intent" in types:
        has_both += 1
    elif "concept" in types:
        has_concept_only += 1
    elif "intent" in types:
        has_intent_only += 1
print(f"  Trajs with concept AND intent: {has_both}")
print(f"  Trajs with concept only:        {has_concept_only}")
print(f"  Trajs with intent only:         {has_intent_only}")

# B7. Per-example_id stats
ex_stats = collections.defaultdict(lambda: {"count": 0, "total_ops": 0})
for r in records:
    eid = r.get("example_id", "unknown")
    ex_stats[eid]["count"] += 1
    ex_stats[eid]["total_ops"] += len(get_ops_list(r))
print(f"\n  Unique example_ids: {len(ex_stats)}")
# Show a few examples with most trajectories
top_ex = sorted(ex_stats.items(), key=lambda x: -x[1]["count"])[:5]
for ex_id, info in top_ex:
    avg_ops = info["total_ops"] / info["count"]
    print(f"    {ex_id}: {info['count']} trajs, avg {avg_ops:.1f} ops/traj")

print("\n" + "=" * 70)
print("Analysis complete.")
