#!/usr/bin/env python3
"""
Simple Telegram Symptom-Checker Bot (single-file)

Requirements (put in requirements.txt):
python-telegram-bot==20.8
reportlab

Notes:
- This script uses only python-telegram-bot for Telegram interactions.
- It uses stdlib csv to load dataset/ CSV files.
- Runs with Application.run_polling().
- Keep instance count = 1 on Render when using polling (or deploy as Background Worker).
"""

import os
import csv
import ast
import logging
from datetime import datetime, timezone
from typing import List, Dict
from io import BytesIO

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# PDF generator
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# -------------- CONFIG --------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # set in Render env
DATA_DIR = os.getenv("DATA_DIR", "dataset")
# Expected CSV filenames inside DATA_DIR:
TRAINING_CSV = os.path.join(DATA_DIR, "Training.csv")
SYMPTOMS_DF_CSV = os.path.join(DATA_DIR, "symptoms_df.csv")
DESCRIPTION_CSV = os.path.join(DATA_DIR, "description.csv")
MEDICATIONS_CSV = os.path.join(DATA_DIR, "medications.csv")
DIETS_CSV = os.path.join(DATA_DIR, "diets.csv")
WORKOUT_CSV = os.path.join(DATA_DIR, "workout_df.csv")
PRECAUTIONS_CSV = os.path.join(DATA_DIR, "precautions_df.csv")
SYMPTOM_SEVERITY_CSV = os.path.join(DATA_DIR, "Symptom-severity.csv")

MIN_CONFIDENCE = 25
TOP_N = 5
EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe bleeding", "unconscious", "loss of consciousness",
    "sudden weakness", "sudden numbness", "slurred speech"
]
# ------------------------------------

# -------------- LOGGING --------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# ------------------------------------

# -------------- UTILITIES --------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def normalize(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = "".join(ch for ch in t if ch.isalnum() or ch.isspace() or ch == "-")
    return " ".join(t.split())

def parse_list_like(cell):
    """Parse cells like "['a','b']" or "a, b, c" into list[str]."""
    if cell is None:
        return []
    if isinstance(cell, (list, tuple)):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

def read_csv_rows(path):
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                # ensure dict
                if not isinstance(r, dict):
                    r = dict(zip(reader.fieldnames or [], list(r)))
                rows.append(r)
    except FileNotFoundError:
        logger.warning("CSV not found: %s", path)
    except Exception as e:
        logger.exception("Error reading CSV %s: %s", path, e)
    return rows
# ---------------------------------------

# -------------- KB LOAD --------------
def build_kb():
    logger.info("Loading CSV datasets from %s", DATA_DIR)
    training_rows = read_csv_rows(TRAINING_CSV)
    symptoms_rows = read_csv_rows(SYMPTOMS_DF_CSV)
    desc_rows = read_csv_rows(DESCRIPTION_CSV)
    meds_rows = read_csv_rows(MEDICATIONS_CSV)
    diets_rows = read_csv_rows(DIETS_CSV)
    workout_rows = read_csv_rows(WORKOUT_CSV)
    prec_rows = read_csv_rows(PRECAUTIONS_CSV)
    severity_rows = read_csv_rows(SYMPTOM_SEVERITY_CSV)

    # Build disease -> symptoms from Training.csv (assumes many symptom columns + final disease column)
    training_map = {}
    for row in training_rows:
        # detect disease column heuristically
        disease_val = None
        for candidate in ("prognosis", "diagnosis", "disease", "diseases", "label"):
            for col in row.keys():
                if col and col.lower() == candidate:
                    disease_val = row.get(col)
                    break
            if disease_val:
                break
        if not disease_val:
            # fallback: last column value
            vals = [v for v in row.values() if v is not None and str(v).strip() != ""]
            disease_val = vals[-1] if vals else None
        if not disease_val:
            continue
        dnorm = normalize(disease_val)
        training_map.setdefault(dnorm, set())
        # assume symptom columns flagged with 1
        for col, val in row.items():
            if col is None:
                continue
            if val is None:
                continue
            sval = str(val).strip().lower()
            if sval in ("1", "true", "yes", "y"):
                training_map[dnorm].add(normalize(col))

    # Build from symptoms_df.csv (Disease, Symptom_1..Symptom_4)
    symptoms_map = {}
    for row in symptoms_rows:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        symptoms_map.setdefault(dnorm, set())
        for key, val in row.items():
            if key and key.lower().startswith("symptom") and val and str(val).strip():
                symptoms_map[dnorm].add(normalize(val))

    # merge maps
    disease_to_symptoms = {}
    for d, sset in training_map.items():
        disease_to_symptoms[d] = set(sset)
    for d, sset in symptoms_map.items():
        disease_to_symptoms.setdefault(d, set()).update(sset)

    # description map
    description_map = {}
    for row in desc_rows:
        dval = row.get("Disease") or row.get("disease") or next(iter(row.values()), None)
        if not dval:
            continue
        dnorm = normalize(dval)
        # join other columns
        parts = []
        for k, v in row.items():
            if k and k.lower() != "disease" and v and str(v).strip():
                parts.append(str(v).strip())
        description_map[dnorm] = " ".join(parts) if parts else ""

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

    # diets: parse list-like strings
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
            # fallback to any non-empty non-disease value
            for k, v in row.items():
                if k and k.lower() != "disease" and v and str(v).strip():
                    diet_cell = v
                    break
        parsed = parse_list_like(diet_cell)
        if parsed:
            diets_map[dnorm] = parsed

    # workouts
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
        precs = []
        for k, v in row.items():
            if k and k.lower().startswith("precaution") and v and str(v).strip():
                precs.append(str(v).strip())
        if not precs:
            for k, v in row.items():
                if k and k.lower() != "disease" and v and str(v).strip():
                    precs.append(str(v).strip())
        if precs:
            precautions_map[dnorm] = precs

    # symptom weights
    symptom_weights = {}
    for row in severity_rows:
        # find symptom & weight columns heuristically
        sym_col = next((k for k in row.keys() if k and "symptom" in k.lower()), None)
        wt_col = next((k for k in row.keys() if k and k.lower() in ("weight", "severity", "value")), None)
        if not sym_col:
            # fallback first column
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
    # normalize weights
    if symptom_weights:
        max_w = max(symptom_weights.values())
        if max_w > 0:
            for k in list(symptom_weights.keys()):
                symptom_weights[k] = symptom_weights[k] / max_w

    # turn sets into lists for JSON safety
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
    logger.info("KB loaded: %d diseases", len(disease_to_symptoms))
    return kb

KB = build_kb()
# --------------------------------------

# -------------- Matching & Scoring --------------
def simple_similarity(a: str, b: str) -> float:
    """Cheap similarity via SequenceMatcher ratio (0..1)."""
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0

def score_diseases(user_symptoms: List[str], severities: List[float]) -> List[Dict]:
    user_norms = [normalize(s) for s in user_symptoms]
    results = []
    for disease, dsym_list in KB["disease_to_symptoms"].items():
        ds_norms = [normalize(s) for s in dsym_list]
        raw = 0.0
        matches = []
        for us_norm, sev in zip(user_norms, severities):
            best = 0.0
            best_sym = None
            for ds in ds_norms:
                # exact/substr check
                if us_norm == ds or us_norm in ds or ds in us_norm:
                    score = 1.0
                else:
                    score = simple_similarity(us_norm, ds)
                if score > best:
                    best = score
                    best_sym = ds
            if best >= 0.3:
                w = KB["symptom_weights"].get(us_norm, KB["symptom_weights"].get(best_sym, 1.0))
                raw += (sev / 5.0) * best * w
                matches.append((best_sym, round(best, 2)))
        max_possible = sum((s / 5.0) for s in severities) or 1.0
        percent = int((raw / max_possible) * 100)
        results.append({"disease": disease, "score_percent": percent, "raw": raw, "matches": matches})
    results.sort(key=lambda r: (-r["score_percent"], -r["raw"]))
    return results
# ------------------------------------------

# -------------- PDF Generation --------------
def generate_pdf_bytes(disease, desc, meds, diets, workouts, precs, symptoms):
    """Return BytesIO containing generated PDF."""
    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"Diagnosis Report — {disease.title()}")
    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {now_iso()}")
    y -= 18
    c.drawString(50, y, f"Symptoms: {', '.join(symptoms)}")
    y -= 20
    if desc:
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Description:"); y -= 14
        c.setFont("Helvetica", 10)
        for line in wrap_text(desc, 90):
            c.drawString(55, y, line); y -= 12
    if meds:
        y -= 6
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Medications:"); y -= 14
        c.setFont("Helvetica", 10)
        for m in meds:
            c.drawString(60, y, f"- {m}"); y -= 12
    if diets:
        y -= 6
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Diet Suggestions:"); y -= 14
        c.setFont("Helvetica", 10)
        for d in diets:
            c.drawString(60, y, f"- {d}"); y -= 12
    if workouts:
        y -= 6
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Workouts:"); y -= 14
        c.setFont("Helvetica", 10)
        c.drawString(60, y, str(workouts)); y -= 14
    if precs:
        y -= 6
        c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Precautions:"); y -= 14
        c.setFont("Helvetica", 10)
        for p in precs:
            c.drawString(60, y, f"- {p}"); y -= 12
    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Disclaimer: This report is educational only and not a professional diagnosis.")
    c.showPage()
    c.save()
    bio.seek(0)
    return bio

def wrap_text(text, width):
    words = text.split()
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
# ------------------------------------------

# -------------- Handlers & Flow --------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data["state"] = "ask_guess"
    await update.message.reply_text(
        "Hello — I'm an educational symptom helper.\n"
        "Do you have a guess what disease you might have? Reply yes/no."
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send symptoms in a message (comma-separated) or follow the interactive prompts. Use /start to begin.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    uid = update.effective_user.id
    state = context.user_data.get("state")

    # if no state, start
    if state is None:
        context.user_data["state"] = "ask_guess"
        await update.message.reply_text("Do you have a guess? (yes/no)")
        return

    # Ask guess?
    if state == "ask_guess":
        if text.lower().startswith("y"):
            context.user_data["state"] = "await_guess_text"
            await update.message.reply_text("What disease do you think you have? (write name)")
        else:
            context.user_data["state"] = "ask_symptoms"
            await update.message.reply_text("Please list your symptoms (comma-separated).")
        return

    if state == "await_guess_text":
        context.user_data["guess"] = text
        context.user_data["state"] = "ask_confidence"
        await update.message.reply_text("How sure are you (0-100)?")
        return

    if state == "ask_confidence":
        try:
            conf = float(text)
        except Exception:
            conf = 50.0
        context.user_data["guess_confidence"] = max(0.0, min(100.0, conf))
        context.user_data["state"] = "ask_symptoms"
        await update.message.reply_text("Now list your symptoms (comma-separated).")
        return

    if state == "ask_symptoms":
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            await update.message.reply_text("Please provide at least one symptom.")
            return
        context.user_data["symptoms"] = parts
        context.user_data["severities"] = []
        context.user_data["sev_index"] = 0
        context.user_data["state"] = "ask_severity"
        await update.message.reply_text(f"Rate severity (1-5) for: {parts[0]}")
        return

    if state == "ask_severity":
        # record
        try:
            v = float(text)
        except Exception:
            v = 3.0
        v = max(1.0, min(5.0, v))
        context.user_data["severities"].append(v)
        idx = context.user_data["sev_index"] + 1
        parts = context.user_data["symptoms"]
        if idx < len(parts):
            context.user_data["sev_index"] = idx
            await update.message.reply_text(f"Rate severity (1-5) for: {parts[idx]}")
            return
        # done collecting severities -> analyze
        results = score_diseases(context.user_data["symptoms"], context.user_data["severities"])
        context.user_data["results"] = results
        # triage check
        joined = " ".join(context.user_data["symptoms"]).lower()
        triage_msg = ""
        if any(k in joined for k in EMERGENCY_KEYWORDS):
            triage_msg = "⚠️ Possible emergency symptom detected — seek urgent care."
        # send top results
        if not results:
            await update.message.reply_text("No matches found in the knowledge base.")
            context.user_data.clear()
            return
        msg_lines = [f"*Top {min(TOP_N, len(results))} possible conditions:*"]
        for r in results[:TOP_N]:
            msg_lines.append(f"- *{r['disease'].title()}* — Confidence: {r['score_percent']}%")
        if results[0]["score_percent"] < MIN_CONFIDENCE:
            msg_lines.append("\n⚠️ Low confidence — consider giving more symptoms.")
        if triage_msg:
            msg_lines.append("\n" + triage_msg)
        # keyboard to get PDF or refine
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("Get PDF report", callback_data="get_pdf"),
            InlineKeyboardButton("Refine", callback_data="refine"),
            InlineKeyboardButton("Done", callback_data="done")
        ]])
        await update.message.reply_text("\n".join(msg_lines), parse_mode=ParseMode.MARKDOWN, reply_markup=kb)
        context.user_data["state"] = "done"
        return

    # done state messages
    if state == "done":
        await update.message.reply_text("If you want to start again, send /start.")
        return

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    if data == "refine":
        context.user_data["state"] = "ask_symptoms"
        await query.edit_message_text("Okay — send additional symptoms or clarify existing ones (comma-separated).")
        return
    if data == "done":
        context.user_data.clear()
        await query.edit_message_text("Okay — session ended. Use /start to run again.")
        return
    if data == "get_pdf":
        res = context.user_data.get("results")
        if not res:
            await query.edit_message_text("No analysis to generate PDF from. Run /start first.")
            return
        top = res[0]
        disease = top["disease"]
        desc = KB["description_map"].get(disease, "")
        meds = KB["meds_map"].get(disease, [])
        diets = KB["diets_map"].get(disease, [])
        workouts = KB["workout_map"].get(disease, "")
        precs = KB["precautions_map"].get(disease, [])
        symptoms = context.user_data.get("symptoms", [])
        bio = generate_pdf_bytes(disease, desc, meds, diets, workouts, precs, symptoms)
        bio.name = f"diagnosis_{disease}.pdf"
        await context.bot.send_document(chat_id=query.message.chat_id, document=InputFile(bio, filename=bio.name), caption="Diagnosis PDF (educational only).")
        await query.edit_message_text("PDF sent. Use /start to analyze another case.")
        context.user_data.clear()
        return

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Exception in handler: %s", context.error)
# ---------------------------------------

# -------------- MAIN --------------
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN environment variable before running.")
    # Build application
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_error_handler(error_handler)

    logger.info("Bot started (polling).")
    app.run_polling()

if __name__ == "__main__":
    main()
