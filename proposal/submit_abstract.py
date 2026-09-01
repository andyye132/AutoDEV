#!/usr/bin/env python3
"""Register the AutoMLR abstract via the OpenReview API (the web button hides after the due date,
but the invitation accepts submissions until its expiration time). Usage:
    pip install openreview-py
    python3 submit_abstract.py            # prompts for OpenReview email + password
"""
import os, sys, getpass
try:
    import openreview
except ImportError:
    sys.exit("run:  pip install openreview-py   (then re-run this script)")

USERNAME = os.environ.get("OPENREVIEW_USERNAME") or input("OpenReview email: ").strip()
PASSWORD = os.environ.get("OPENREVIEW_PASSWORD") or getpass.getpass("OpenReview password: ")

TITLE = ("Do Discovered Game-Testing Objectives Generalize? Autonomous Discovery of "
         "Exploration Rewards for Finding Injected Defects in Unseen Games")
KEYWORDS = ["autonomous research agents", "automated game testing", "reinforcement learning",
            "exploration", "intrinsic motivation", "LLM-driven program evolution", "bug detection"]
TLDR = ("An autonomous code-evolution loop discovers an exploration reward for finding injected "
        "game defects that transfers to unseen games and bug classes.")
ABSTRACT = """Learning-based game testers rely on hand-designed exploration objectives. We ask whether an autonomous code-evolution loop can discover a testing objective that generalizes — to games it never saw and to defect classes it was never searched on. We build a GPU-vectorized suite of 2D physics games (Kinetix) with five classes of programmatically injected defects (collision holes, out-of-bounds, unintended shortcuts, reward exploits, physics anomalies) and witness-state oracles, freeze a sealed held-out split of games and defect classes, and let an LLM-driven evolutionary loop propose intrinsic-reward programs whose fitness is expected steps-to-first-defect on the search set. Qualifying result (agent-decisive): the discovered objective reduces steps-to-first-defect relative to the best tuned hand-designed baseline (count-based, RND, ICM, curiosity-proximity) on held-out games and a held-out defect class, while objectives evolved against task return transfer to fewer held-out classes; a distilled core of the discovered objective retains most of the gain. The objective, the transfer analysis, and the write-up of Part 1 are produced by the autonomous loop; Part 2 documents the harness, prompts, human interventions, and the recomputation of every reported number from logs."""

client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net", username=USERNAME, password=PASSWORD)
profile = None
try:
    profile = getattr(client, "profile", None)
except Exception: pass
if not profile:
    try:
        found = client.search_profiles(emails=[USERNAME]); profile = found.get(USERNAME) if isinstance(found, dict) else (found[0] if found else None)
    except Exception as e: print("search_profiles failed:", e)
if not profile:
    try: profile = client.get_profile(input("Your OpenReview profile ID (from the URL at openreview.net/profile, e.g. ~Andy_Ye1): ").strip())
    except Exception as e: sys.exit(f"could not load profile: {e}")
pid = profile.id
names = (getattr(profile, "content", None) or {}).get("names", [{}]) or [{}]
fullname = names[0].get("fullname") or (names[0].get("first", "") + " " + names[0].get("last", "")).strip() or "Andy Ye"
print(f"Logged in as {fullname} ({pid})")

note = openreview.api.Note(content={
    "title": {"value": TITLE},
    "authors": {"value": [fullname]},
    "authorids": {"value": [pid]},
    "keywords": {"value": KEYWORDS},
    "TLDR": {"value": TLDR},
    "abstract": {"value": ABSTRACT},
    "serve_as_reviewer": {"value": [pid]},
})
edit = client.post_note_edit(invitation="NeurIPS.cc/2026/Workshop/AutoMLR/-/Submission",
                             signatures=[pid], note=note)
nid = edit["note"]["id"]
print("SUBMITTED. Note id:", nid)
print("View/edit it at: https://openreview.net/forum?id=" + nid)
