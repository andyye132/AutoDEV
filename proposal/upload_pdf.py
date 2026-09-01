"""Attach the paper PDF (and optionally an updated abstract/title) to the AutoMLR submission you already registered on
OpenReview. Run this yourself (it needs your OpenReview login); it edits the existing note through the same Submission
invitation used for the abstract, which accepts edits until its expiration time.

    python proposal/upload_pdf.py paper/main.pdf [--abstract paper/abstract.txt] [--title "..."]
"""
import os, sys, argparse, getpass
import openreview

ap = argparse.ArgumentParser(); ap.add_argument("pdf"); ap.add_argument("--abstract", default=""); ap.add_argument("--title", default="")
ap.add_argument("--venue", default="NeurIPS.cc/2026/Workshop/AutoMLR"); a = ap.parse_args()
USERNAME = os.environ.get("OPENREVIEW_USERNAME") or input("OpenReview username (email): ").strip()
PASSWORD = os.environ.get("OPENREVIEW_PASSWORD") or getpass.getpass("OpenReview password: ")
client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net", username=USERNAME, password=PASSWORD)
inv = f"{a.venue}/-/Submission"
notes = client.get_notes(invitation=inv, tauthor=True)      # notes you authored under this invitation
if not notes:
    sys.exit("No submission of yours found under " + inv + " — check the venue id or log in with the account that submitted.")
note = notes[0]
print(f"Found note {note.id} (number {note.number}): {note.content['title']['value']}")
url = client.put_attachment(a.pdf, inv, "pdf")
content = {"pdf": {"value": url}}
if a.abstract: content["abstract"] = {"value": open(a.abstract).read().strip()}
if a.title: content["title"] = {"value": a.title}
edit = client.post_note_edit(invitation=inv, signatures=[f"{a.venue}/Submission{note.number}/Authors"],
                             note=openreview.api.Note(id=note.id, content=content))
print("Uploaded. Edit id:", edit.get("id") if isinstance(edit, dict) else getattr(edit, "id", edit))
print("Verify at https://openreview.net/forum?id=" + note.id)
