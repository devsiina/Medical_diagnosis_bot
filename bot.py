#!/usr/bin/env python3
"""
telegram_symptom_bot.py

Telegram bot version of the symptom checker using CSV datasets stored in dataset/.

Requires:
- pip install pyTelegramBotAPI reportlab

Expected dataset files (placed in ./dataset/):
- Training.csv
- symptoms_df.csv
- description.csv
- medications.csv
- diets.csv
- workout_df.csv
- precautions_df.csv
- Symptom-severity.csv

This script:
- Interacts via Telegram messages
- Handles conversation flow with user states
- Scores diseases and provides remedies
- Generates PDF report on request
- Logs sessions to local_session_logs.csv

DISCLAIMER: educational/testing only. NOT medical advice.

For deployment:
- Set TELEGRAM_TOKEN environment variable
- Include dataset/ folder
- For Heroku: Create requirements.txt with pyTelegramBotAPI and reportlab
- Procfile: worker: python telegram_symptom_bot.py
"""

import os
import csv
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
import json
import telebot
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ----------------- CONFIG -----------------
DATA_DIR = "dataset"
TRAINING_CSV = os.path.join(DATA_DIR, "Training.csv")
SYMPTOMS_DF_CSV = os.path.join(DATA_DIR, "symptoms_df.csv")
DESCRIPTION_CSV = os.path.join(DATA_DIR, "description.csv")
MEDICATIONS_CSV = os.path.join(DATA_DIR, "medications.csv")
DIETS_CSV = os.path.join(DATA_DIR, "diets.csv")
WORKOUT_CSV = os.path.join(DATA_DIR, "workout_df.csv")
PRECAUTIONS_CSV = os.path.join(DATA_DIR, "precautions_df.csv")
SYMPTOM_SEVERITY_CSV = os.path.join(DATA_DIR, "Symptom-severity.csv")

SESSION_LOG = "local_session_logs.csv"

# Matching thresholds / params
FUZZY_THRESHOLD = 0.50   # minimum fuzzy ratio considered some match (0..1)
FUZZY_STRONG = 0.70      # considered a good match
GUESS_BOOST_THRESHOLD = 0.80  # if user's guess matches disease >= this, we boost
MAX_GUESS_BOOST_FACTOR = 0.20  # up to +20% relative boost depending on confidence

EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe bleeding", "unconscious", "loss of consciousness",
    "sudden weakness", "sudden numbness", "slurred speech"
]
TOP_N = 6
# ------------------------------------------

bot = telebot.TeleBot(os.getenv("TELEGRAM_TOKEN"))
user_states = {}

def normalize(text):
    if text is None:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s\-]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def fuzzy(a, b):
    a = normalize(a)
    b = normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def load_training_mapping(path):
    if not os.path.exists(path):
        print(f"[warn] Training CSV not found: {path}")
        return {}

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
        if not rows:
            return {}
        headers = rows[0]
        possible_disease_names = set([h.lower() for h in headers]) & set(["prognosis", "diagnosis", "disease", "diseases", "label", "target"])
        if possible_disease_names:
            disease_col = headers[[h.lower() for h in headers].index(next(iter(possible_disease_names)))]
            disease_idx = headers.index(disease_col)
        else:
            disease_idx = len(headers) - 1
            disease_col = headers[disease_idx]

        symptom_cols = [h for i, h in enumerate(headers) if i != disease_idx]

        disease_map = {}
        for row in rows[1:]:
            if len(row) <= disease_idx:
                continue
            disease_raw = row[disease_idx]
            if not disease_raw:
                continue
            dnorm = normalize(disease_raw)
            if dnorm not in disease_map:
                disease_map[dnorm] = set()
            for i, col in enumerate(headers):
                if i == disease_idx:
                    continue
                val = row[i] if i < len(row) else ""
                if str(val).strip() in ("1", "1.0", "true", "True", "TRUE", "yes", "y", "Y"):
                    disease_map[dnorm].add(normalize(col))
        return disease_map

def load_symptoms_df(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_d = row.get("Disease") or row.get("disease") or row.get("Disease".lower())
            if not raw_d:
                continue
            dnorm = normalize(raw_d)
            if dnorm not in mapping:
                mapping[dnorm] = set()
            for key, val in row.items():
                if key is None:
                    continue
                if key.lower().startswith("symptom") and val and val.strip():
                    mapping[dnorm].add(normalize(val))
    return mapping

def load_symptom_weights(path):
    weights = {}
    if not os.path.exists(path):
        return weights
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = [h.lower() for h in reader.fieldnames] if reader.fieldnames else []
        sym_key = None
        weight_key = None
        for h in reader.fieldnames or []:
            hl = h.lower()
            if hl in ("symptoms", "symptom", "symptom_name", "symptom_text"):
                sym_key = h
            if hl in ("weight", "severity", "value"):
                weight_key = h
        if not sym_key and reader.fieldnames:
            sym_key = reader.fieldnames[0]
        if not weight_key and len(reader.fieldnames or []) > 1:
            weight_key = reader.fieldnames[1]
        for row in reader:
            sym = row.get(sym_key) if sym_key else None
            w = row.get(weight_key) if weight_key else None
            if not sym:
                continue
            try:
                wf = float(w) if w not in (None, "") else 1.0
            except Exception:
                wf = 1.0
            weights[normalize(sym)] = wf
    if weights:
        max_w = max(weights.values())
        if max_w > 0:
            for k in list(weights.keys()):
                weights[k] = weights[k] / max_w
    return weights

def load_text_map(path, key_col="Disease", value_col=None):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return mapping
        for row in reader:
            raw_d = row.get(key_col) or row.get(key_col.lower()) or row.get(key_col.capitalize())
            if not raw_d:
                raw_d = next(iter(row.values()))
            dnorm = normalize(raw_d)
            if value_col:
                val = row.get(value_col, "")
            else:
                parts = []
                for k, v in row.items():
                    if not v:
                        continue
                    if k == (key_col if key_col in row else list(row.keys())[0]):
                        continue
                    if v.strip():
                        parts.append(f"{k}: {v.strip()}")
                val = "; ".join(parts)
            mapping[dnorm] = val or ""
    return mapping

def aggregate_kb():
    disease_map = {}
    tmap = load_training_mapping(TRAINING_CSV)
    for d, syms in tmap.items():
        disease_map.setdefault(d, set()).update(syms)
    s2 = load_symptoms_df(SYMPTOMS_DF_CSV)
    for d, syms in s2.items():
        disease_map.setdefault(d, set()).update(syms)
    return disease_map

def emergency_detect(symptoms_norm, guess_norm=""):
    text = " ".join(symptoms_norm + ([guess_norm] if guess_norm else []))
    for ek in EMERGENCY_KEYWORDS:
        if ek in text:
            return ek
    return None

def score_all(user_symptoms_norm, user_severities, disease_map, weights_map, user_guess_norm=None, user_guess_conf=0):
    max_possible = sum((s / 5.0) * 1.0 for s in user_severities) or 1.0
    results = []
    for disease, dsymset in disease_map.items():
        raw = 0.0
        matches = []
        for us_norm, sev in zip(user_symptoms_norm, user_severities):
            best_ratio = 0.0
            best_ds = None
            for ds in dsymset:
                r = fuzzy(us_norm, ds)
                if r > best_ratio:
                    best_ratio = r
                    best_ds = ds
            if best_ratio >= FUZZY_THRESHOLD:
                sym_weight = weights_map.get(us_norm, weights_map.get(best_ds, 1.0))
                contrib = (sev / 5.0) * best_ratio * sym_weight
                raw += contrib
                matches.append((best_ds or "?", best_ratio, sev, sym_weight))
            else:
                if best_ratio >= (FUZZY_THRESHOLD * 0.8):
                    sym_weight = weights_map.get(us_norm, weights_map.get(best_ds, 1.0))
                    contrib = (sev / 5.0) * best_ratio * sym_weight * 0.4
                    raw += contrib
                    matches.append((best_ds or "?", best_ratio, sev, sym_weight))
        percent = int((raw / max_possible) * 100)
        guess_boost_info = None
        if user_guess_norm:
            gscore = fuzzy(user_guess_norm, disease)
            if gscore >= GUESS_BOOST_THRESHOLD:
                boost_factor = (user_guess_conf / 100.0) * MAX_GUESS_BOOST_FACTOR
                boost_amount = int(percent * boost_factor)
                percent = min(100, percent + boost_amount)
                guess_boost_info = (gscore, boost_amount)
        results.append({
            "disease": disease,
            "raw_score": raw,
            "score_percent": percent,
            "matches": matches,
            "guess_boost": guess_boost_info
        })
    results.sort(key=lambda x: (-x["score_percent"], -x["raw_score"]))
    return results

def safe_title(s):
    return s.replace("_", " ").title()

def parse_list_val(val):
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            parsed = json.loads(val.replace("'", '"'))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return val

def save_session_log(entry):
    header = ["timestamp","guess","guess_confidence","symptoms","severities","top_disease","top_score","top_matches"]
    exists = os.path.exists(SESSION_LOG)
    with open(SESSION_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            entry.get("timestamp",""),
            entry.get("guess",""),
            entry.get("guess_confidence",""),
            json.dumps(entry.get("symptoms",[]), ensure_ascii=False),
            json.dumps(entry.get("severities",[]), ensure_ascii=False),
            entry.get("top_disease",""),
            entry.get("top_score",""),
            json.dumps(entry.get("top_matches",[]), ensure_ascii=False)
        ])

def ask_next_severity(user_id):
    index = user_states[user_id].get('current_severity_index', 0)
    symptoms = user_states[user_id]['data'].get('symptoms', [])
    if index >= len(symptoms):
        # Check emergency
        data = user_states[user_id]['data']
        user_guess = data.get('guess', '')
        user_guess_norm = normalize(user_guess)
        sym_norm = data.get('symptoms_norm', [])
        ek = emergency_detect(sym_norm, user_guess_norm)
        if ek:
            bot.send_message(user_id, "\n!!! EMERGENCY WARNING !!!")
            bot.send_message(user_id, f"The term '{ek}' appears in your symptoms/guess and is considered a red flag.")
            bot.send_message(user_id, "If you are experiencing a medical emergency (chest pain, severe difficulty breathing, severe bleeding, unconsciousness), please seek immediate medical attention.")
            bot.send_message(user_id, "Continue with the test output anyway (not recommended)? [y/n]")
            user_states[user_id]['state'] = 'confirm_proceed'
        else:
            compute_and_send(user_id)
        return
    bot.send_message(user_id, f"Severity for '{symptoms[index]}' (1-5)")
    user_states[user_id]['state'] = 'get_severity'

def compute_and_send(user_id):
    data = user_states[user_id]['data']
    user_guess = data.get('guess', '')
    user_guess_conf = data.get('conf', 0)
    user_symptoms = data.get('symptoms', [])
    user_symptoms_norm = data.get('symptoms_norm', [])
    severities = data.get('severities', [])
    user_guess_norm = normalize(user_guess) if user_guess else None

    results = score_all(user_symptoms_norm, severities, disease_map, weights_map, user_guess_norm, float(user_guess_conf or 0))

    if not results:
        bot.send_message(user_id, "No diseases scored. Check dataset files.")
        del user_states[user_id]
        return

    # Send analysis results
    text = "=== Analysis Results ===\n\n"
    top = results[:TOP_N]
    for idx, r in enumerate(top, start=1):
        text += f"{idx}. {safe_title(r['disease'])} — Confidence: {r['score_percent']}%\n"
        if r['guess_boost']:
            gscore, boost_amount = r['guess_boost']
            text += f"   (User guess matched this disease (score {gscore:.2f}) -> boost +{boost_amount} pts)\n"
        if r['matches']:
            matches_tr = []
            for m in r['matches']:
                ds_name, ratio, sev, w = m
                matches_tr.append(f"{ds_name} (match {int(ratio*100)}%, sev {int(sev)}, w {w:.2f})")
            text += "   Matched symptoms: " + "; ".join(matches_tr) + "\n"
        else:
            text += "   Matched symptoms: none significant\n"
        text += "\n"

    bot.send_message(user_id, text)

    # Top suggestion details
    top0 = top[0]
    disease_key = top0['disease']
    display_name = safe_title(disease_key)
    desc = desc_map.get(disease_key, "")
    meds = parse_list_val(meds_map.get(disease_key, ""))
    diet = parse_list_val(diets_map.get(disease_key, ""))
    workout = parse_list_val(workout_map.get(disease_key, ""))
    prec = prec_map.get(disease_key, "")

    detail_text = f"=== Top suggestion details ===\nDisease: {display_name}\n"
    if desc:
        detail_text += "\nDescription:\n" + desc + "\n"
    if meds:
        detail_text += "\nMedications / Recommendations:\n"
        if isinstance(meds, list):
            detail_text += "\n".join("- " + item for item in meds) + "\n"
        else:
            detail_text += meds + "\n"
    if diet:
        detail_text += "\nDiet / Nutrition suggestions:\n"
        if isinstance(diet, list):
            detail_text += "\n".join("- " + item for item in diet) + "\n"
        else:
            detail_text += diet + "\n"
    if workout:
        detail_text += "\nWorkout / Exercise suggestions:\n"
        if isinstance(workout, list):
            detail_text += "\n".join("- " + item for item in workout) + "\n"
        else:
            detail_text += workout + "\n"
    if prec:
        detail_text += "\nPrecautions / Preventive measures:\n" + prec + "\n"
    if not any((desc, meds, diet, workout, prec)):
        detail_text += "\nNo additional info found in dataset for this disease.\n"
    detail_text += "\nNote: All remedies/information are shown verbatim from your dataset (if present).\nThis is educational only — not a substitute for professional care.\n"

    bot.send_message(user_id, detail_text)

    # Log session
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "guess": user_guess,
        "guess_confidence": user_guess_conf,
        "symptoms": user_symptoms,
        "severities": severities,
        "top_disease": top0['disease'],
        "top_score": top0['score_percent'],
        "top_matches": [(m[0], round(m[1],2), int(m[2])) for m in top0['matches']]
    }
    try:
        save_session_log(log_entry)
    except Exception as e:
        print("[warn] Could not write session log:", e)

    # Store results for PDF
    user_states[user_id]['results'] = {
        'display_name': display_name,
        'desc': desc,
        'meds': meds,
        'diet': diet,
        'workout': workout,
        'prec': prec
    }

    # Ask for PDF
    bot.send_message(user_id, "Do you want to generate a PDF report? [y/n]")
    user_states[user_id]['state'] = 'ask_pdf'

def generate_pdf_and_send(user_id):
    results = user_states[user_id]['results']
    pdf_file = f"diagnosis_{user_id}.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    y = 750
    c.drawString(100, y, "Diagnosis Report")
    y -= 20
    c.drawString(100, y, f"Disease: {results['display_name']}")
    y -= 30

    if results['desc']:
        c.drawString(100, y, "Description:")
        y -= 20
        lines = results['desc'].split('\n')
        for line in lines:
            c.drawString(100, y, line)
            y -= 15
            if y < 50:
                c.showPage()
                y = 750
        y -= 10

    if results['meds']:
        c.drawString(100, y, "Medications / Recommendations:")
        y -= 20
        if isinstance(results['meds'], list):
            for item in results['meds']:
                c.drawString(100, y, "- " + item)
                y -= 15
                if y < 50:
                    c.showPage()
                    y = 750
        else:
            c.drawString(100, y, results['meds'])
            y -= 15
        y -= 10

    if results['diet']:
        c.drawString(100, y, "Diet / Nutrition suggestions:")
        y -= 20
        if isinstance(results['diet'], list):
            for item in results['diet']:
                c.drawString(100, y, "- " + item)
                y -= 15
                if y < 50:
                    c.showPage()
                    y = 750
        else:
            c.drawString(100, y, results['diet'])
            y -= 15
        y -= 10

    if results['workout']:
        c.drawString(100, y, "Workout / Exercise suggestions:")
        y -= 20
        if isinstance(results['workout'], list):
            for item in results['workout']:
                c.drawString(100, y, "- " + item)
                y -= 15
                if y < 50:
                    c.showPage()
                    y = 750
        else:
            c.drawString(100, y, results['workout'])
            y -= 15
        y -= 10

    if results['prec']:
        c.drawString(100, y, "Precautions / Preventive measures:")
        y -= 20
        lines = results['prec'].split('\n')
        for line in lines:
            c.drawString(100, y, line)
            y -= 15
            if y < 50:
                c.showPage()
                y = 750

    c.drawString(100, 50, "Note: This is educational only — not a substitute for professional care.")
    c.save()

    with open(pdf_file, 'rb') as f:
        bot.send_document(user_id, f)
    os.remove(pdf_file)
    bot.send_message(user_id, "PDF report sent.")
    del user_states[user_id]

@bot.message_handler(commands=['start'])
def start_handler(message):
    user_id = message.chat.id
    bot.send_message(user_id, "=== Symptom Checker Bot on Telegram ===\nDISCLAIMER: Educational/testing only. NOT medical advice.")
    bot.send_message(user_id, "Do you have a guess for what disease you might have? [y/n]")
    user_states[user_id] = {'state': 'ask_guess', 'data': {}}

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.chat.id
    if user_id not in user_states:
        bot.send_message(user_id, "Please use /start to begin.")
        return
    state = user_states[user_id]['state']
    text = message.text.strip().lower()

    if state == 'ask_guess':
        if text in ('y', 'yes'):
            bot.send_message(user_id, "Enter your guessed disease name:")
            user_states[user_id]['state'] = 'get_guess_name'
        elif text in ('n', 'no'):
            bot.send_message(user_id, "Enter your symptoms (comma-separated). Example: fever, cough, headache")
            user_states[user_id]['state'] = 'get_symptoms'
        else:
            bot.send_message(user_id, "Please answer y or n.")

    elif state == 'get_guess_name':
        user_states[user_id]['data']['guess'] = message.text.strip()
        bot.send_message(user_id, "How sure are you? (0-100)")
        user_states[user_id]['state'] = 'get_guess_conf'

    elif state == 'get_guess_conf':
        try:
            conf = float(message.text.strip())
            if 0 <= conf <= 100:
                user_states[user_id]['data']['conf'] = conf
                bot.send_message(user_id, "Enter your symptoms (comma-separated). Example: fever, cough, headache")
                user_states[user_id]['state'] = 'get_symptoms'
            else:
                raise ValueError
        except:
            bot.send_message(user_id, "Enter a number between 0-100.")

    elif state == 'get_symptoms':
        raw = message.text.strip()
        symptoms = [s.strip() for s in re.split(r",|;|\n", raw) if s.strip()]
        if not symptoms:
            bot.send_message(user_id, "Please enter at least one symptom.")
            return
        user_states[user_id]['data']['symptoms'] = symptoms
        user_states[user_id]['data']['symptoms_norm'] = [normalize(s) for s in symptoms]
        user_states[user_id]['data']['severities'] = []
        user_states[user_id]['current_severity_index'] = 0
        bot.send_message(user_id, "For each symptom provide severity (1 = mild, 5 = severe).")
        ask_next_severity(user_id)

    elif state == 'get_severity':
        try:
            v = float(message.text.strip())
            if 1 <= v <= 5:
                user_states[user_id]['data']['severities'].append(v)
                user_states[user_id]['current_severity_index'] += 1
                ask_next_severity(user_id)
            else:
                raise ValueError
        except:
            bot.send_message(user_id, "Enter a number 1-5.")

    elif state == 'confirm_proceed':
        if text in ('y', 'yes'):
            compute_and_send(user_id)
        elif text in ('n', 'no'):
            bot.send_message(user_id, "Stopping. Seek help.")
            del user_states[user_id]
        else:
            bot.send_message(user_id, "Please answer y or n.")

    elif state == 'ask_pdf':
        if text in ('y', 'yes'):
            generate_pdf_and_send(user_id)
        elif text in ('n', 'no'):
            bot.send_message(user_id, "Okay.")
            del user_states[user_id]
        else:
            bot.send_message(user_id, "Please answer y or n.")

if __name__ == "__main__":
    print("Loading datasets from:", DATA_DIR)
    disease_map = aggregate_kb()
    if not disease_map:
        print("No disease mapping found. Please ensure Training.csv or symptoms_df.csv exist in 'dataset/'")
        exit(1)
    weights_map = load_symptom_weights(SYMPTOM_SEVERITY_CSV)
    desc_map = load_text_map(DESCRIPTION_CSV, key_col="Disease", value_col=None)
    meds_map = load_text_map(MEDICATIONS_CSV, key_col="Disease", value_col="medication")
    diets_map = load_text_map(DIETS_CSV, key_col="Disease", value_col="Diet")
    workout_map = load_text_map(WORKOUT_CSV, key_col="Disease", value_col="Workout")
    prec_map = load_text_map(PRECAUTIONS_CSV, key_col="Disease", value_col=None)
    print(f"Loaded {len(disease_map)} diseases in KB.")
    print("Starting Telegram bot polling...")
    bot.polling()
