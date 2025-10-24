#!/usr/bin/env python3
"""
Simple Telegram Symptom Checker (single-file, stdlib only)

- No external packages required.
- Uses Telegram Bot HTTP API via urllib (long-polling getUpdates).
- Loads CSV datasets from dataset/ (pure csv module).
- No PDF generation. No python-telegram-bot usage.
- Set environment variable: TELEGRAM_TOKEN
"""

import os
import sys
import time
import csv
import ast
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import List

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    print("ERROR: Set TELEGRAM_TOKEN environment variable.", file=sys.stderr)
    sys.exit(1)

API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
DATA_DIR = os.getenv("DATA_DIR", "dataset")
POLL_TIMEOUT = int(os.getenv("POLL_TIMEOUT", "30"))  # seconds for getUpdates timeout
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "0.5"))  # sleep between getUpdates on errors
# Expected dataset file paths inside DATA_DIR:
TRAINING_CSV = os.path.join(DATA_DIR, "Training.csv")
SYMPTOMS_DF_CSV = os.path.join(DATA_DIR, "symptoms_df.csv")
DESCRIPTION_CSV = os.path.join(DATA_DIR, "description.csv")
MEDICATIONS_CSV = os.path.join(DATA_DIR, "medications.csv")
DIETS_CSV = os.path.join(DATA_DIR, "diets.csv")
WORKOUT_CSV = os.path.join(DATA_DIR, "workout_df.csv")
PRECAUTIONS_CSV = os.path.join(DATA_DIR, "precautions_df.csv")
SYMPTOM_SEVERITY_CSV = os.path.join(DATA_DIR, "Symptom-severity.csv")
# --------------------------------

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("simple_symptom_bot")
# --------------------------------

# ---------- Utilities ----------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def normalize(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    # keep alnum, space, hyphen
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace() or ch == "-")
    return " ".join(t.split())

def parse_list_like(cell):
    """Parse "['a','b']" or "a, b, c" into list[str]."""
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
    # fallback: comma/semicolon split
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return parts

def read_csv_rows(path):
    rows = []
    if not os.path.exists(path):
        logger.warning("CSV not found: %s", path)
        return rows
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if not isinstance(r, dict):
                    r = dict(zip(reader.fieldnames or [], list(r)))
                rows.append(r)
    except Exception as e:
        logger.exception("Failed to read CSV %s: %s", path, e)
    return rows

# ---------- KB building ----------
def build_kb():
    logger.info("Loading KB files from %s", DATA_DIR)
    training = read_csv_rows(TRAINING_CSV)
    symptoms_df = read_csv_rows(SYMPTOMS_DF_CSV)
    desc_rows = read_csv_rows(DESCRIPTION_CSV)
    meds_rows = read_csv_rows(MEDICATIONS_CSV)
    diets_rows = read_csv_rows(DIETS_CSV)
    workout_rows = read_csv_rows(WORKOUT_CSV)
    prec_rows = read_csv_rows(PRECAUTIONS_CSV)
    severity_rows = read_csv_rows(SYMPTOM_SEVERITY_CSV)

    # Training -> disease -> symptoms (assumes many symptom columns flagged 1)
    training_map = {}
    for row in training:
        # detect disease value
        disease_val = None
        for candidate in ("prognosis", "diagnosis", "disease", "label"):
            for col in row.keys():
                if col and col.lower() == candidate:
                    disease_val = row.get(col)
                    break
            if disease_val:
                break
        if not disease_val:
            vals = [v for v in row.values() if v is not None and str(v).strip() != ""]
            disease_val = vals[-1] if vals else None
        if not disease_val:
            continue
        dnorm = normalize(disease_val)
        training_map.setdefault(dnorm, set())
        for col, val in row.items():
            if col is None:
                continue
            if val is None:
                continue
            sval = str(val).strip().lower()
            if sval in ("1", "true", "yes", "y"):
                training_map[dnorm].add(normalize(col))

    # symptoms_df mapping
    symptoms_map = {}
    for row in symptoms_df:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        symptoms_map.setdefault(dnorm, set())
        for k, v in row.items():
            if k and k.lower().startswith("symptom") and v and str(v).strip():
                symptoms_map[dnorm].add(normalize(v))

    # merge
    disease_to_symptoms = {}
    for d, s in training_map.items():
        disease_to_symptoms[d] = set(s)
    for d, s in symptoms_map.items():
        disease_to_symptoms.setdefault(d, set()).update(s)

    # descriptions
    description_map = {}
    for row in desc_rows:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        parts = []
        for k, v in row.items():
            if k and k.lower() != "disease" and v and str(v).strip():
                parts.append(str(v).strip())
        description_map[dnorm] = " ".join(parts)

    # meds
    meds_map = {}
    for row in meds_rows:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        med = None
        for k, v in row.items():
            if v and "med" in (k or "").lower():
                med = v
                break
        if med:
            meds_map.setdefault(dnorm, []).append(str(med).strip())

    # diets: parse list-like
    diets_map = {}
    for row in diets_rows:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        diet_cell = None
        for k, v in row.items():
            if v and "diet" in (k or "").lower():
                diet_cell = v
                break
        if not diet_cell:
            for k, v in row.items():
                if k and k.lower() != "disease" and v and str(v).strip():
                    diet_cell = v
                    break
        parsed = parse_list_like(diet_cell)
        if parsed:
            diets_map[dnorm] = parsed

    # workout
    workout_map = {}
    for row in workout_rows:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        wval = None
        for k, v in row.items():
            if v and ("work" in (k or "").lower() or "exercise" in (k or "").lower()):
                wval = v
                break
        if wval:
            workout_map[dnorm] = str(wval).strip()

    # precautions
    precautions_map = {}
    for row in prec_rows:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        p_list = []
        for k, v in row.items():
            if k and k.lower().startswith("precaution") and v and str(v).strip():
                p_list.append(str(v).strip())
        if not p_list:
            for k, v in row.items():
                if k and k.lower() != "disease" and v and str(v).strip():
                    p_list.append(str(v).strip())
        if p_list:
            precautions_map[dnorm] = p_list

    # symptom weights
    symptom_weights = {}
    for row in severity_rows:
        sym_col = next((k for k in row.keys() if k and "symptom" in k.lower()), None)
        wt_col = next((k for k in row.keys() if k and k.lower() in ("weight", "severity", "value")), None)
        if not sym_col:
            sym_col = next(iter(row.keys()), None)
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
    if symptom_weights:
        max_w = max(symptom_weights.values())
        if max_w > 0:
            for k in list(symptom_weights.keys()):
                symptom_weights[k] = symptom_weights[k] / max_w

    # convert sets to lists
    disease_to_symptoms = {k: list(v) for k, v in disease_to_symptoms.items()}

    kb = {
        "disease_to_symptoms": disease_to_symptoms,
        "description_map": description_map,
        "meds_map": meds_map,
        "diets_map": diets_map,
        "workout_map": workout_map,
        "precautions_map": precautions_map,
        "symptom_weights": symptom_weights,
    }
    logger.info("KB built: %d diseases, %d symptom weights", len(disease_to_symptoms), len(symptom_weights))
    return kb

KB = build_kb()

# ---------- Telegram helpers (urllib) ----------
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
    with urllib.request.urlopen(url, timeout=POLL_TIMEOUT + 10) as resp:
        return json.load(resp)

def send_message(chat_id, text, parse_mode=None):
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        api_post_json("sendMessage", payload)
    except Exception as e:
        logger.exception("sendMessage failed: %s", e)

# ---------- Scoring ----------
def simple_ratio(a: str, b: str) -> float:
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0

def score_diseases(user_symptoms: List[str], severities: List[float]):
    user_norms = [normalize(s) for s in user_symptoms]
    results = []
    for disease, dsym_list in KB["disease_to_symptoms"].items():
        ds_norms = [normalize(x) for x in dsym_list]
        raw = 0.0
        matches = []
        for us, sev in zip(user_norms, severities):
            best = 0.0
            best_sym = None
            for ds in ds_norms:
                if us == ds or us in ds or ds in us:
                    score = 1.0
                else:
                    score = simple_ratio(us, ds)
                if score > best:
                    best = score; best_sym = ds
            if best >= 0.3:
                weight = KB["symptom_weights"].get(us, KB["symptom_weights"].get(best_sym, 1.0))
                raw += (sev / 5.0) * best * weight
                matches.append((best_sym, round(best,2)))
        max_possible = sum((s/5.0) for s in severities) or 1.0
        percent = int((raw / max_possible) * 100)
        results.append({"disease": disease, "score": percent, "raw": raw, "matches": matches})
    results.sort(key=lambda r: (-r["score"], -r["raw"]))
    return results

# ---------- Conversation state ----------
# minimal memory: user_id -> {step, guess, confidence, symptoms[], severities[]}
SESSIONS = {}

def reset_session(uid):
    SESSIONS[uid] = {"step":"start", "guess":"", "confidence":0.0, "symptoms":[], "severities":[]}

# ---------- Update processing ----------
def process_message(chat_id, user_id, text):
    text = (text or "").strip()
    if user_id not in SESSIONS:
        reset_session(user_id)
        SESSIONS[user_id]["step"] = "ask_guess"
        send_message(chat_id, "Hi — welcome. Do you have a guess for your illness? (yes/no)")
        return

    session = SESSIONS[user_id]
    step = session["step"]

    if step == "ask_guess":
        if text.lower().startswith("y"):
            session["step"] = "ask_guess_text"
            send_message(chat_id, "What disease do you think you have? (name)")
        else:
            session["step"] = "ask_symptoms"
            send_message(chat_id, "Okay. Please list your symptoms, separated by commas.")
        return

    if step == "ask_guess_text":
        session["guess"] = text
        session["step"] = "ask_confidence"
        send_message(chat_id, "How sure are you? Enter a percentage 0-100.")
        return

    if step == "ask_confidence":
        try:
            v = float(text)
            v = max(0.0, min(100.0, v))
        except Exception:
            v = 50.0
        session["confidence"] = v
        session["step"] = "ask_symptoms"
        send_message(chat_id, "Now list your symptoms (comma-separated).")
        return

    if step == "ask_symptoms":
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            send_message(chat_id, "Please provide at least one symptom.")
            return
        session["symptoms"] = parts
        session["severities"] = []
        session["step"] = "ask_severity"
        session["sev_index"] = 0
        send_message(chat_id, f"Rate severity (1-5) for: {parts[0]}")
        return

    if step == "ask_severity":
        try:
            v = float(text)
            if v < 1: v = 1.0
            if v > 5: v = 5.0
        except Exception:
            v = 3.0
        session["severities"].append(v)
        idx = session.get("sev_index", 0) + 1
        if idx < len(session["symptoms"]):
            session["sev_index"] = idx
            send_message(chat_id, f"Rate severity (1-5) for: {session['symptoms'][idx]}")
            return
        # done -> analyze
        results = score_diseases(session["symptoms"], session["severities"])
        if not results:
            send_message(chat_id, "Sorry — no matches found in knowledge base.")
            reset_session(user_id)
            return
        # reply top 5
        topn = results[:5]
        lines = ["*Possible conditions (prototype):*"]
        for r in topn:
            lines.append(f"- *{r['disease'].title()}* — Confidence: {r['score']}%")
        # remedies and extra info for top 1
        top0 = topn[0]
        dkey = top0["disease"]
        desc = KB["description_map"].get(dkey, "")
        meds = KB["meds_map"].get(dkey, [])
        diets = KB["diets_map"].get(dkey, [])
        workouts = KB["workout_map"].get(dkey, "")
        precs = KB["precautions_map"].get(dkey, [])
        lines.append("\n*Top suggestion details:*")
        if desc:
            lines.append(f"_Description:_ {desc}")
        if meds:
            lines.append(f"_Medications:_ {', '.join(meds)}")
        if diets:
            lines.append(f"_Diet:_ {', '.join(diets)}")
        if workouts:
            lines.append(f"_Workout:_ {workouts}")
        if precs:
            lines.append(f"_Precautions:_ {', '.join(precs)}")
        # triage check
        joined = " ".join(session["symptoms"]).lower()
        if any(k in joined for k in ("chest pain", "difficulty breathing", "shortness of breath", "severe bleeding")):
            lines.append("\n⚠️ *Emergency warning:* Seek immediate medical help if symptoms are severe.")
        send_message(chat_id, "\n".join(lines), parse_mode="Markdown")
        reset_session(user_id)
        return

    # fallback
    send_message(chat_id, "I didn't understand. Send /start to begin again.")
    reset_session(user_id)

# ---------- Polling loop ----------
def polling_loop():
    offset = None
    logger.info("Starting long-polling loop (getUpdates).")
    while True:
        try:
            params = {"timeout": POLL_TIMEOUT, "limit": 10}
            if offset:
                params["offset"] = offset
            resp = api_get("getUpdates", params=params)
            if not resp.get("ok"):
                logger.warning("getUpdates returned not ok: %s", resp)
                time.sleep(POLL_INTERVAL)
                continue
            updates = resp.get("result", []) or []
            for u in updates:
                offset = u["update_id"] + 1
                # handle message
                if "message" in u:
                    m = u["message"]
                    chat = m.get("chat", {})
                    chat_id = chat.get("id")
                    user = m.get("from", {})
                    user_id = user.get("id")
                    text = m.get("text")
                    # Commands
                    if text and text.strip().lower() == "/start":
                        reset_session(user_id)
                        send_message(chat_id, "Welcome — do you have a guess for your illness? (yes/no)")
                        SESSIONS[user_id]["step"] = "ask_guess"
                        continue
                    process_message(chat_id, user_id, text)
                # ignore other update types for simplicity
        except urllib.error.HTTPError as e:
            logger.exception("HTTPError during getUpdates: %s", e)
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.exception("Polling loop exception: %s", e)
            time.sleep(POLL_INTERVAL)

# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        polling_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt).")
        sys.exit(0)
