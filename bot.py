#!/usr/bin/env python3
"""
Telegram Symptom-Checker Bot (HTTP polling, stdlib)
- Uses Telegram Bot API directly (no python-telegram-bot).
- Loads CSV datasets from dataset/ (pure stdlib csv).
- Generates PDF reports with reportlab and sends via sendDocument.
- Use as a single-process Background Worker on Render (replicas=1).
"""

import os
import sys
import time
import csv
import ast
import json
import logging
import mimetypes
import urllib.parse
import urllib.request
import io
from datetime import datetime, timezone

# PDF generation (external package)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    print("ERROR: TELEGRAM_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)

API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
KB_DIR = os.getenv("KB_DIR", "dataset")
KB_PATHS = {
    "training": os.path.join(KB_DIR, "Training.csv"),
    "symptoms_df": os.path.join(KB_DIR, "symptoms_df.csv"),
    "description": os.path.join(KB_DIR, "description.csv"),
    "medications": os.path.join(KB_DIR, "medications.csv"),
    "diets": os.path.join(KB_DIR, "diets.csv"),
    "workout": os.path.join(KB_DIR, "workout_df.csv"),
    "precautions": os.path.join(KB_DIR, "precautions_df.csv"),
    "severity": os.path.join(KB_DIR, "Symptom-severity.csv"),
}
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.0"))  # seconds between getUpdates
LOG_FILE = os.getenv("LOG_FILE", None)  # optional file path
# --------------------------------

# ---------- Logging ----------
logger = logging.getLogger("symptom_bot")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
h = logging.StreamHandler(sys.stdout)
h.setFormatter(fmt)
logger.addHandler(h)
if LOG_FILE:
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# ---------- Utilities ----------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def normalize(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch == "-")
    s = " ".join(s.split())
    return s

def parse_possible_list_field(cell):
    """Parse "['a','b']" or "a, b, c" into list of strings."""
    if cell is None:
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # fallback: split by comma/semicolon
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

# ---------- CSV loaders (pure stdlib) ----------
def load_csv_rows(path):
    rows = []
    if not os.path.exists(path):
        logger.warning("CSV not found: %s", path)
        return rows
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                # ensure r is dict
                if not isinstance(r, dict):
                    r = dict(zip(reader.fieldnames or [], list(r)))
                rows.append(r)
    except Exception as e:
        logger.exception("Failed to load CSV %s: %s", path, e)
    return rows

def build_kb():
    logger.info("Loading KB files from %s", KB_DIR)
    kb = {}
    for k, p in KB_PATHS.items():
        kb[k] = load_csv_rows(p)
        logger.info("Loaded %d rows from %s", len(kb[k]), p)
    # Build convenient maps
    # training mapping: disease_norm -> set(symptom_norm) from Training.csv (assumes presence of many symptom columns)
    training_map = {}
    for row in kb.get("training", []):
        # detect disease column heuristically
        disease = None
        for candidate in ("prognosis", "diagnosis", "disease", "label"):
            if candidate in (c.lower() for c in row.keys()):
                disease = row.get(next(c for c in row.keys() if c.lower() == candidate))
                break
        if not disease:
            # fallback: last non-empty value
            vals = [v for v in row.values() if v is not None and str(v).strip() != ""]
            if vals:
                disease = vals[-1]
        if not disease:
            continue
        dnorm = normalize(disease)
        if dnorm not in training_map:
            training_map[dnorm] = set()
        # symptoms columns: assume all columns except disease column; if header value '1' => symptom present
        for col, val in row.items():
            if col is None:
                continue
            # skip if this column's value equals the disease value (heuristic)
            if str(val).strip().lower() == str(disease).strip().lower():
                continue
            if str(val).strip() in ("1", "1.0", "true", "yes"):
                training_map[dnorm].add(normalize(col))
    # symptoms_df mapping: Disease -> symptom columns (Symptom_1..)
    symptoms_map = {}
    for row in kb.get("symptoms_df", []):
        # find disease value
        dval = None
        for key in row.keys():
            if key and key.lower() == "disease":
                dval = row.get(key)
                break
        if not dval:
            # try first column
            dval = next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        if dnorm not in symptoms_map:
            symptoms_map[dnorm] = set()
        for col, val in row.items():
            if col and col.lower().startswith("symptom") and val and str(val).strip():
                symptoms_map[dnorm].add(normalize(val))
    # merge training_map and symptoms_map
    disease_to_symptoms = {}
    for d, sset in training_map.items():
        disease_to_symptoms[d] = set(sset)
    for d, sset in symptoms_map.items():
        disease_to_symptoms.setdefault(d, set()).update(sset)
    # parse description, meds, diets, workouts, precautions maps keyed by normalized disease
    description_map = {}
    for row in kb.get("description", []):
        # look for Disease column
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        description_map[normalize(dval)] = " ".join([str(v).strip() for k, v in row.items() if k and k.lower() != "disease" and v])
    meds_map = {}
    for row in kb.get("medications", []):
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        med = ""
        # find a meds column
        for k, v in row.items():
            if v and "med" in (k or "").lower():
                med = v
                break
        meds_map.setdefault(normalize(dval), []).append(med) if med else None
    diets_map = {}
    for row in kb.get("diets", []):
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        # find diet column heuristically
        diet_val = ""
        for k, v in row.items():
            if v and "diet" in (k or "").lower():
                diet_val = v
                break
        if not diet_val:
            # try any non-empty value except disease
            for k, v in row.items():
                if v and (k.lower() != "disease"):
                    diet_val = v
                    break
        parsed = parse_possible_list_field(diet_val)
        if parsed:
            diets_map[normalize(dval)] = parsed
    workout_map = {}
    for row in kb.get("workout", []):
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        w = ""
        for k, v in row.items():
            if v and ("work" in (k or "").lower() or "workout" in (k or "").lower() or "exercise" in (k or "").lower()):
                w = v
                break
        if w:
            workout_map[normalize(dval)] = str(w)
    precautions_map = {}
    for row in kb.get("precautions", []):
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        p_list = []
        for k, v in row.items():
            if k and k.lower().startswith("precaution") and v and str(v).strip():
                p_list.append(str(v).strip())
        if not p_list:
            # try any non-empty columns except disease
            for k, v in row.items():
                if k and k.lower() != "disease" and v and str(v).strip():
                    p_list.append(str(v).strip())
        if p_list:
            precautions_map[normalize(dval)] = p_list
    # symptom weights
    symptom_weights = {}
    for row in kb.get("severity", []):
        # detect symptom & weight columns
        sym_col = None
        wt_col = None
        for k in row.keys():
            if k and "symptom" in k.lower():
                sym_col = k
            if k and (k.lower() in ("weight", "severity", "value")):
                wt_col = k
        # fallback
        if not sym_col and row and len(row) >= 1:
            sym_col = next(iter(row.keys()))
        if not wt_col and row and len(row) >= 2:
            wt_col = list(row.keys())[1]
        if not sym_col:
            continue
        s = row.get(sym_col)
        w = row.get(wt_col) if wt_col else None
        if s:
            try:
                wf = float(w) if w not in (None, "") else 1.0
            except Exception:
                wf = 1.0
            symptom_weights[normalize(s)] = wf
    # normalize weights
    if symptom_weights:
        m = max(symptom_weights.values())
        if m > 0:
            for k in list(symptom_weights.keys()):
                symptom_weights[k] = symptom_weights[k] / m
    logger.info("KB built: %d diseases, %d symptom weights", len(disease_to_symptoms), len(symptom_weights))
    return {
        "disease_to_symptoms": {k: list(v) for k, v in disease_to_symptoms.items()},
        "description_map": description_map,
        "meds_map": meds_map,
        "diets_map": diets_map,
        "workout_map": workout_map,
        "precautions_map": precautions_map,
        "symptom_weights": symptom_weights
    }

KB = build_kb()

# ---------- Telegram HTTP helpers (stdlib) ----------
def api_post_json(method, data):
    url = f"{API_BASE}/{method}"
    data_bytes = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)

def api_get(method, params=None):
    url = f"{API_BASE}/{method}"
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)

def send_message(chat_id, text, parse_mode=None):
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        api_post_json("sendMessage", payload)
    except Exception as e:
        logger.exception("send_message failed: %s", e)

def send_document(chat_id, file_path, filename=None):
    """Send a file via multipart/form-data using urllib (stdlib)."""
    url = f"{API_BASE}/sendDocument"
    boundary = "----WebKitFormBoundary" + str(int(time.time() * 1000))
    if not filename:
        filename = os.path.basename(file_path)
    # read file bytes
    with open(file_path, "rb") as fh:
        file_bytes = fh.read()
    # build multipart body
    crlf = b"\r\n"
    body = []
    # chat_id field
    body.append(b"--" + boundary.encode("utf-8"))
    body.append(b'Content-Disposition: form-data; name="chat_id"')
    body.append(b"")
    body.append(str(chat_id).encode("utf-8"))
    # document file field
    body.append(b"--" + boundary.encode("utf-8"))
    disposition = f'Content-Disposition: form-data; name="document"; filename="{filename}"'
    body.append(disposition.encode("utf-8"))
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body.append(f"Content-Type: {ctype}".encode("utf-8"))
    body.append(b"")
    body.append(file_bytes)
    # end
    body.append(b"--" + boundary.encode("utf-8") + b"--")
    body.append(b"")
    body_bytes = crlf.join(body)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body_bytes))
    }
    req = urllib.request.Request(url, data=body_bytes, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    except Exception as e:
        logger.exception("send_document failed: %s", e)
        return None

# ---------- Simple scoring & analysis ----------
def score_diseases(user_symptoms, user_severities):
    """
    Very simple scoring:
    - For each disease, count how many normalized symptoms match disease symptom list (substring/fuzzy naive)
    - Weight by severity and symptom weight if available
    - Return sorted list
    """
    user_norms = [normalize(s) for s in user_symptoms]
    results = []
    for disease, dsym_list in KB["disease_to_symptoms"].items():
        dsym_norms = [normalize(s) for s in dsym_list]
        raw = 0.0
        matches = []
        for us, sev in zip(user_norms, user_severities):
            best = 0.0
            best_sym = None
            for ds in dsym_norms:
                # simple substring / fuzzy via ratio
                if us == ds or us in ds or ds in us:
                    score = 1.0
                else:
                    # use simple character ratio
                    score = char_ratio(us, ds)
                if score > best:
                    best = score
                    best_sym = ds
            if best > 0.3:
                weight = KB["symptom_weights"].get(us, KB["symptom_weights"].get(best_sym, 1.0))
                raw += (sev / 5.0) * best * weight
                matches.append((best_sym, best))
        # normalize by number of user symptoms
        max_possible = sum((s/5.0) for s in user_severities) or 1.0
        percent = int((raw / max_possible) * 100)
        results.append({"disease": disease, "score": percent, "raw": raw, "matches": matches})
    results.sort(key=lambda r: (-r["score"], -r["raw"]))
    return results

def char_ratio(a, b):
    # cheap similarity: longest common subsequence ratio via SequenceMatcher
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        # fallback
        return 0.0

# ---------- PDF generation ----------
def create_pdf(disease, desc, meds, diets, workouts, precautions, symptoms):
    fname = f"diagnosis_{disease.replace(' ', '_')}_{int(time.time())}.pdf"
    c = canvas.Canvas(fname, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Diagnosis Report — {disease}")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {now_iso()}")
    y -= 20
    c.drawString(50, y, f"Symptoms: {', '.join(symptoms)}")
    y -= 30
    if desc:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Description:"); y -= 16
        c.setFont("Helvetica", 10); text = c.beginText(50, y)
        for line in split_text(desc, 80):
            text.textLine(line); y -= 12
        c.drawText(text)
        y = text.getY() - 10
    if meds:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Medications:"); y -= 16
        c.setFont("Helvetica", 10)
        for m in meds:
            c.drawString(60, y, f"- {m}"); y -= 12
    if diets:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Diet suggestions:"); y -= 16
        c.setFont("Helvetica", 10)
        for d in diets:
            c.drawString(60, y, f"- {d}"); y -= 12
    if workouts:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Workouts:"); y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(60, y, str(workouts)); y -= 14
    if precautions:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Precautions:"); y -= 16
        c.setFont("Helvetica", 10)
        for p in precautions:
            c.drawString(60, y, f"- {p}"); y -= 12
    y -= 20
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: educational only — not a medical diagnosis.")
    c.save()
    return fname

def split_text(s, width):
    words = s.split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 <= width:
            cur.append(w); cur_len += len(w) + 1
        else:
            lines.append(" ".join(cur)); cur = [w]; cur_len = len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return lines

# ---------- Conversation management ----------
# Minimal in-memory state: user_id -> dict with keys: step, guess, confidence, symptoms (list), severities (list)
SESSIONS = {}

def reset_session(user_id):
    SESSIONS[user_id] = {"step": "start", "guess": "", "confidence": 0, "symptoms": [], "severities": []}

# ---------- Update processing ----------
def handle_update(u):
    """Process a single Telegram update dict."""
    if "message" not in u:
        return
    m = u["message"]
    chat = m.get("chat", {})
    chat_id = chat.get("id")
    user = m.get("from", {})
    user_id = user.get("id")
    text = m.get("text", "")
    if not text:
        return
    text_stripped = text.strip()
    if user_id not in SESSIONS:
        reset_session(user_id)
        send_message(chat_id, "Hi — welcome to the Symptom Checker (prototype). Do you have a guess about your illness? (yes/no)")
        SESSIONS[user_id]["step"] = "awaiting_guess_answer"
        return
    session = SESSIONS[user_id]
    step = session["step"]

    # flow
    if step == "awaiting_guess_answer":
        if text_stripped.lower().startswith("y"):
            session["step"] = "ask_guess"
            send_message(chat_id, "Okay — what disease do you think you have?")
            return
        else:
            session["step"] = "ask_symptoms"
            send_message(chat_id, "Please list your symptoms, separated by commas (e.g., fever, cough).")
            return
    if step == "ask_guess":
        session["guess"] = text_stripped
        session["step"] = "ask_confidence"
        send_message(chat_id, "How sure are you? Enter a percentage 0-100.")
        return
    if step == "ask_confidence":
        try:
            conf = float(text_stripped)
            session["confidence"] = max(0, min(100, conf))
        except Exception:
            session["confidence"] = 50.0
        session["step"] = "ask_symptoms"
        send_message(chat_id, "Now, please list your symptoms (comma-separated).")
        return
    if step == "ask_symptoms":
        parts = [p.strip() for p in text_stripped.split(",") if p.strip()]
        session["symptoms"] = parts
        session["severities"] = []
        if not parts:
            send_message(chat_id, "Please provide at least one symptom.")
            return
        session["step"] = "ask_severity_index"
        session["severity_index"] = 0
        send_message(chat_id, f"Rate severity (1-5) for: {parts[0]}")
        return
    if step == "ask_severity_index":
        # record severity for current symptom
        try:
            v = float(text_stripped)
            if v < 1: v = 1
            if v > 5: v = 5
        except Exception:
            v = 3.0
        idx = session.get("severity_index", 0)
        session["severities"].append(v)
        idx += 1
        if idx < len(session["symptoms"]):
            session["severity_index"] = idx
            send_message(chat_id, f"Rate severity (1-5) for: {session['symptoms'][idx]}")
            return
        # all severities collected -> analyze
        session["step"] = "analysis_done"
        results = score_diseases(session["symptoms"], session["severities"])
        # build reply
        if not results:
            send_message(chat_id, "No matches found in KB.")
            reset_session(user_id)
            return
        # prepare top results text
        top_texts = []
        topn = results[:5]
        for r in topn:
            top_texts.append(f"{r['disease'].title()} — {r['score']}%")
        reply = "Top possible conditions:\n" + "\n".join(top_texts)
        reply += "\n\nReply 'pdf' to receive a PDF report of the top result, or 'no' to finish."
        send_message(chat_id, reply)
        # store top result in session
        session["top_results"] = results
        return
    if step == "analysis_done":
        if text_stripped.lower() in ("pdf", "yes", "y"):
            top = session.get("top_results", [])
            if not top:
                send_message(chat_id, "No analysis to generate PDF from.")
                reset_session(user_id)
                return
            top0 = top[0]
            disease = top0["disease"]
            desc = KB["description_map"].get(disease, "")
            meds = KB["meds_map"].get(disease, [])
            diets = KB["diets_map"].get(disease, [])
            workouts = KB["workout_map"].get(disease, "")
            precs = KB["precautions_map"].get(disease, [])
            pdf_path = create_pdf(disease, desc, meds, diets, workouts, precs, session.get("symptoms", []))
            send_message(chat_id, f"Sending PDF report for {disease} — this is educational only.")
            send_document(chat_id, pdf_path, filename=os.path.basename(pdf_path))
            try:
                os.remove(pdf_path)
            except Exception:
                pass
            reset_session(user_id)
            return
        else:
            send_message(chat_id, "Okay — finished. If you want another analysis, send any message.")
            reset_session(user_id)
            return

# ---------- Polling loop ----------
def main_loop():
    offset = None
    logger.info("Starting polling loop...")
    while True:
        try:
            params = {"timeout": 30}
            if offset:
                params["offset"] = offset
            resp = api_get("getUpdates", params=params)
            if not resp.get("ok"):
                logger.warning("getUpdates not ok: %s", resp)
                time.sleep(POLL_INTERVAL)
                continue
            updates = resp.get("result", [])
            for u in updates:
                # process update
                handle_update(u)
                offset = u["update_id"] + 1
        except Exception as e:
            logger.exception("Polling loop error: %s", e)
            time.sleep(5)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt).")
