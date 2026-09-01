"""The evaluation protocol, frozen and hashed so that every result file says which rules produced it.
Change a field only with a new PROTOCOL_VERSION; analysis scripts refuse to mix hashes."""
import hashlib, json
PROTOCOL = dict(
    version="2026-08-30a",
    harness="one writer per archive slot per step; lineage log; verified hits (Kinetix); conditional reset; trash log entry",
    fitness=dict(split="reachable search split", seeds=[0, 1, 2], workers=128, steps=1024, verify_hits=True),
    evaluation=dict(splits=["search", "heldout_levels_instances", "heldout_class_instances"], seeds=[100, 101, 102, 103, 104], workers=128, steps=2048, verify_hits=True),
    exclusions="instances unwitnessable by construction or witnessed within 8 no-op steps are dropped from denominators (results/structural.json)",
    score="1 - min(first_hit, budget)/budget, first_hit in worker-steps; detected = first_hit < budget",
)
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:12]
