#!/usr/bin/env python3
"""Sample specific failure examples from each category."""
import json

INPUT = r"f:\cognifold dataset\CogniFold\dataset\output\musique_trajectories.jsonl"
records = []
with open(INPUT, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
N = len(records)
print(f"Loaded {N} records\n")

def extract_node_id(op):
    data = op.get("data") or {}
    return (data.get("event_id") or data.get("concept_id") or
            data.get("intent_id") or data.get("id") or data.get("node_id") or
            op.get("event_id") or op.get("node_id"))

# --- PARSE ERROR with non-empty response (the 1 edge case) ---
print("=" * 70)
print("FAILURE 1: Parse error BUT non-empty raw_response")
print("=" * 70)
for r in records:
    pe = r["output"].get("parse_error")
    rr = (r["output"].get("raw_response") or "").strip()
    if pe and rr:
        print(f"  trajectory_id: {r['trajectory_id']}")
        print(f"  parse_error: {pe}")
        print(f"  raw_response (first 600 chars):")
        print(f"  {rr[:600]}")
        print()
        break

# --- ORPHAN NODE ---
print("=" * 70)
print("FAILURE 2: The 1 orphan node trajectory")
print("=" * 70)
for r in records:
    ops = r.get("output", {}).get("parsed_plan") or {}
    ops_list = ops.get("operations", []) or []
    edge_sources = set()
    edge_targets = set()
    non_event_nodes = []
    for op in ops_list:
        if op.get("op") == "ADD_EDGE":
            edge_sources.add(op.get("source_id"))
            edge_targets.add(op.get("target_id"))
        elif op.get("op") == "ADD_NODE" and op.get("node_type") != "event":
            nid = extract_node_id(op)
            if nid and nid not in edge_sources and nid not in edge_targets:
                non_event_nodes.append((nid, op.get("node_type"), op.get("data")))
    if non_event_nodes:
        print(f"  trajectory_id: {r['trajectory_id']}")
        print(f"  example_id: {r['example_id']}")
        for nid, ntype, ndata in non_event_nodes:
            print(f"  orphan node: id={nid}, type={ntype}")
            print(f"  data keys: {list(ndata.keys()) if ndata else 'None'}")
        print(f"  All operations ({len(ops_list)}):")
        for op in ops_list:
            print(f"    op={op.get('op',''):15s} node_type={str(op.get('node_type','')):10s} source={str(op.get('source_id','')):20s} target={str(op.get('target_id','')):20s} edge_type={str(op.get('edge_type','')):15s}")
        print()
        break

# --- 0 ops / missing event node ---
print("=" * 70)
print("FAILURE 3: 0-ops trajectory (missing event ADD_NODE)")
print("=" * 70)
count = 0
for r in records:
    ops = r.get("output", {}).get("parsed_plan") or {}
    ops_list = ops.get("operations", []) or []
    if len(ops_list) == 0:
        count += 1
        if count <= 3:
            print(f"  [{count}] trajectory_id: {r['trajectory_id']}")
            print(f"      parse_error: {r['output'].get('parse_error')}")
            rr = (r['output'].get('raw_response') or "")
            print(f"      raw_response length: {len(rr)}")
            if rr:
                print(f"      raw_response preview: {rr[:300]}")
            else:
                print(f"      raw_response: EMPTY")
            print()
print(f"  ... total 0-ops: {count}")

# --- 21+ ops trajectory ---
print("=" * 70)
print("FAILURE 4: The 21+ ops trajectory")
print("=" * 70)
for r in records:
    ops = r.get("output", {}).get("parsed_plan") or {}
    ops_list = ops.get("operations", []) or []
    if len(ops_list) > 20:
        print(f"  trajectory_id: {r['trajectory_id']}")
        print(f"  example_id: {r['example_id']}")
        print(f"  ops count: {len(ops_list)}")
        print(f"  reasoning: {(ops.get('reasoning') or '')[:200]}")
        print(f"  Operations:")
        for op in ops_list:
            print(f"    op={op.get('op',''):15s} node_type={str(op.get('node_type','')):10s} edge_type={str(op.get('edge_type','')):12s}")
        break

# --- REMOVE_EDGE example ---
print()
print("=" * 70)
print("FAILURE 5: A trajectory with REMOVE_EDGE")
print("=" * 70)
for r in records:
    ops = r.get("output", {}).get("parsed_plan") or {}
    ops_list = ops.get("operations", []) or []
    if any(o.get("op") == "REMOVE_EDGE" for o in ops_list):
        print(f"  trajectory_id: {r['trajectory_id']}")
        print(f"  example_id: {r['example_id']}")
        print(f"  reasoning: {(ops.get('reasoning') or '')[:300]}")
        print(f"  All operations:")
        for op in ops_list:
            print(f"    {json.dumps(op, default=str)[:250]}")
        break

# --- MERGE_NODES example ---
print()
print("=" * 70)
print("FAILURE 6: A trajectory with MERGE_NODES")
print("=" * 70)
for r in records:
    ops = r.get("output", {}).get("parsed_plan") or {}
    ops_list = ops.get("operations", []) or []
    if any(o.get("op") == "MERGE_NODES" for o in ops_list):
        print(f"  trajectory_id: {r['trajectory_id']}")
        print(f"  example_id: {r['example_id']}")
        for op in ops_list:
            if op.get("op") == "MERGE_NODES":
                print(f"  MERGE op: {json.dumps(op, default=str)[:400]}")
        break

# --- CONCEPT + INTENT trajectories ---
print()
print("=" * 70)
print("FAILURE 7: Trajectories with both concept AND intent")
print("=" * 70)
count = 0
for r in records:
    ops = r.get("output", {}).get("parsed_plan") or {}
    ops_list = ops.get("operations", []) or []
    types = set()
    for op in ops_list:
        if op.get("op") == "ADD_NODE":
            types.add(op.get("node_type"))
    if "concept" in types and "intent" in types:
        count += 1
        print(f"  [{count}] trajectory_id: {r['trajectory_id']}")
        print(f"      example_id: {r['example_id']}")
        print(f"      reasoning: {(ops.get('reasoning') or '')[:300]}")
        for op in ops_list:
            if op.get("op") == "ADD_NODE":
                nt = op.get("node_type")
                data = op.get("data") or {}
                if nt == "concept":
                    print(f"      CONCEPT: {data.get('title', data.get('concept_id',''))} -> grounded_in={op.get('grounded_in')}")
                elif nt == "intent":
                    print(f"      INTENT:  {data.get('title', data.get('intent_id',''))} -> grounded_in={op.get('grounded_in')}")
        print()

# --- INTENT-only trajectory ---
print("=" * 70)
print("FAILURE 8: Trajectory with intent only (no concept)")
print("=" * 70)
for r in records:
    ops = r.get("output", {}).get("parsed_plan") or {}
    ops_list = ops.get("operations", []) or []
    types = set()
    for op in ops_list:
        if op.get("op") == "ADD_NODE":
            types.add(op.get("node_type"))
    if "intent" in types and "concept" not in types:
        print(f"  trajectory_id: {r['trajectory_id']}")
        print(f"  example_id: {r['example_id']}")
        print(f"  reasoning: {(ops.get('reasoning') or '')[:400]}")
        for op in ops_list:
            print(f"    {json.dumps(op, default=str)[:300]}")
        break

# --- Symbolic actions example ---
print()
print("=" * 70)
print("EXAMPLE: Trajectory with rich symbolic_actions")
print("=" * 70)
for r in records:
    sa = (r.get("output", {}).get("parsed_plan") or {})
    sa_list = sa.get("symbolic_actions") or []
    if len(sa_list) >= 10:
        print(f"  trajectory_id: {r['trajectory_id']}")
        print(f"  example_id: {r['example_id']}")
        print(f"  event_title: {r['input']['event']['title']}")
        print(f"  symbolic_actions ({len(sa_list)}):")
        for a in sa_list[:5]:
            print(f"    {json.dumps(a)}")
        print(f"    ... ({len(sa_list)-5} more)")
        break

print()
print("Done.")
