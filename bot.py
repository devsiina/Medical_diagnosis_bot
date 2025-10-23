#!/usr/bin/env python3
"""
Deployment-ready Telegram Symptom-Checker (polling mode)

Features:
- Loads dataset/ files (Training.csv, symptoms_df.csv, Symptom-severity.csv,
  description.csv, medications.csv, diets.csv, workout_df.csv, precautions_df.csv).
- Normalizes and merges KB into disease -> symptoms mapping.
- Fuzzy matching (rapidfuzz if installed, else difflib).
- Timezone-aware logging (UTC).
- Generates a PDF report of the diagnosis on user request and sends it.
- SQLite logging of queries.

IMPORTANT: Educational only — NOT medical advice.
"""
import os
import re
import csv
import json
import sqlite3
import logging
import tempfile
import ast
from datetime import datetime, timezone
from typing import List, Dict, Tuple
from collections import defaultdict

import pandas as pd

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# Try optional dependencies
try:
    from rapidfuzz.fuzz import ratio as rf_ratio
    RAPIDFUZZ = True
except Exception:
    import difflib
    RAPIDFUZZ = False

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, ListFlowable, ListItem

# Try dotenv for local `.env`
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
# default KB paths (comma-separated env var or fallback to dataset/ file names)
KB_PATHS = [p.strip() for p in os.getenv("KB_PATHS", "dataset/Training.csv,dataset/symptoms_df.csv").split(",")]
LOG_DB = os.getenv("LOG_DB", "symptom_bot_logs.db")
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "30"))
TOP_N = int(os.getenv("TOP_N", "5"))
# --------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe bleeding", "unconscious", "loss of consciousness",
    "sudden weakness", "sudden numbness", "slurred speech"
]

# ---------- Normalization & fuzzy ----------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s\-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def fuzzy_ratio(a: str, b: str) -> int:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n or not b_n:
        return 0
    if RAPIDFUZZ:
        try:
            return int(rf_ratio(a_n, b_n))
        except Exception:
            pass
    return int(difflib.SequenceMatcher(None, a_n, b_n).ratio() * 100)

# ---------- KB loading helpers ----------
def parse_possible_list_field(cell_value):
    """
    Accepts:
     - string representation of Python list: "['a','b']" -> returns list
     - comma/semicolon separated string -> returns list
     - actual list -> returns as-is
    """
    if cell_value is None:
        return []
    if isinstance(cell_value, (list, tuple)):
        return [str(x).strip() for x in cell_value if str(x).strip()]
    s = str(cell_value).strip()
    if not s:
        return []
    # try ast.literal_eval if it looks like a Python list
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # fallback: comma/semicolon split
    parts = [p.strip() for p in re.split(r",|;", s) if p.strip()]
    return parts

def load_kb_files(paths: List[str]):
    """
    Load dataset files and return:
      - disease_to_symptoms: dict disease_norm -> list of symptom_norm
      - meta maps for remedies/info: description_map, meds_map, diets_map, workout_map, precautions_map
      - symptom_weights: dict symptom_norm -> float (0..1)
    """
    disease_to_symptoms = defaultdict(list)
    description_map = {}
    meds_map = {}
    diets_map = {}
    workout_map = {}
    precautions_map = {}
    symptom_weights = {}

    # Load Training.csv -> many symptom columns with 0/1 and last col as Disease (heuristic)
    for p in paths:
        p = p.strip()
        if not p or not os.path.exists(p):
            logger.warning("KB file not found or empty: %s", p)
            continue
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext in (".csv",):
                df = pd.read_csv(p)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                # non-tabular fallback: try read_csv
                df = pd.read_csv(p)
        except Exception as e:
            logger.exception("Failed to read file %s: %s", p, e)
            continue

        # Heuristics based on filename
        name = os.path.basename(p).lower()

        if "training" in name:
            # Find disease column; fallback to last
            columns = list(df.columns)
            disease_col = None
            for candidate in ("prognosis", "diagnosis", "disease", "diseases", "label", "target"):
                if candidate in [c.lower() for c in columns]:
                    disease_col = columns[[c.lower() for c in columns].index(candidate)]
                    break
            if disease_col is None:
                disease_col = columns[-1]
            symptom_cols = [c for c in columns if c != disease_col]
            for _, row in df.iterrows():
                disease_raw = row.get(disease_col)
                if not disease_raw or str(disease_raw).strip() == "":
                    continue
                dnorm = normalize_text(disease_raw)
                for sc in symptom_cols:
                    val = row.get(sc)
                    if str(val).strip() in ("1", "1.0", "True", "true", "YES", "Yes", "Y"):
                        disease_to_symptoms[dnorm].append(normalize_text(sc))
        elif "symptom" in name:
            # symptoms_df.csv: columns Disease, Symptom_1..Symptom_4
            # Loop rows and collect symptom columns
            for _, row in df.iterrows():
                raw_d = row.get("Disease") or row.get("disease") or next(iter(row.values()))
                if not raw_d:
                    continue
                dnorm = normalize_text(raw_d)
                for col in df.columns:
                    if str(col).lower().startswith("symptom"):
                        val = row.get(col)
                        if val and str(val).strip():
                            disease_to_symptoms[dnorm].append(normalize_text(val))
        elif "symptom-severity" in name or "symptom" in name and "severity" in name:
            # Symptom-severity.csv
            # Expect columns Symptoms, weight (or similar)
            # We'll parse below after the loop by saving the df
            pass
        elif "description" in name:
            for _, row in df.iterrows():
                raw_d = row.get("Disease") or row.get("disease") or next(iter(row.values()))
                if not raw_d:
                    continue
                dnorm = normalize_text(raw_d)
                # Collect textual description from all other columns
                parts = []
                for c in df.columns:
                    if c and c != "Disease" and c != "disease":
                        v = row.get(c)
                        if v and str(v).strip():
                            parts.append(str(v).strip())
                description_map[dnorm] = " ".join(parts).strip()
        elif "medicat" in name or "medication" in name:
            for _, row in df.iterrows():
                raw_d = row.get("Disease") or row.get("disease") or next(iter(row.values()))
                if not raw_d:
                    continue
                dnorm = normalize_text(raw_d)
                # try a medication column
                med_col = None
                for c in df.columns:
                    if "med" in c.lower():
                        med_col = c
                        break
                if med_col:
                    meds_map[dnorm] = str(row.get(med_col) or "").strip()
        elif "diet" in name:
            for _, row in df.iterrows():
                raw_d = row.get("Disease") or row.get("disease") or next(iter(row.values()))
                if not raw_d:
                    continue
                dnorm = normalize_text(raw_d)
                # diet cell may be a Python-list-string -> parse with parse_possible_list_field
                diet_col = None
                for c in df.columns:
                    if "diet" in c.lower():
                        diet_col = c
                        break
                if diet_col:
                    diets_map[dnorm] = parse_possible_list_field(row.get(diet_col))
        elif "workout" in name or "exercise" in name:
            for _, row in df.iterrows():
                raw_d = row.get("Disease") or row.get("disease") or next(iter(row.values()))
                if not raw_d:
                    continue
                dnorm = normalize_text(raw_d)
                wcol = None
                for c in df.columns:
                    if "work" in c.lower() or "exercise" in c.lower():
                        wcol = c
                        break
                if wcol:
                    workout_map[dnorm] = str(row.get(wcol) or "").strip()
        elif "precaution" in name:
            for _, row in df.iterrows():
                raw_d = row.get("Disease") or row.get("disease") or next(iter(row.values()))
                if not raw_d:
                    continue
                dnorm = normalize_text(raw_d)
                # collect precaution columns
                precs = []
                for c in df.columns:
                    if "precaution" in c.lower() or "precaution" in str(c).lower() or "precaution" in (str(row.get(c) or "")).lower():
                        v = row.get(c)
                        if v and str(v).strip():
                            precs.append(str(v).strip())
                # fallback: collect any non-empty columns except Disease
                if not precs:
                    for c in df.columns:
                        if c and c != "Disease":
                            v = row.get(c)
                            if v and str(v).strip():
                                precs.append(str(v).strip())
                precautions_map[dnorm] = precs
        elif "symptom" in name and "severity" in name:
            # redundant, handled below
            pass
        # We'll also check if this file contains symptom weight column (heuristic)
        # (Handled outside loop for reliability)

    # Second pass for files that might not be named clearly (e.g., Symptom-severity.csv)
    # Look for any file in paths that contains columns like 'Symptoms'/'weight'
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)
        except Exception:
            continue
        cols_l = [c.lower() for c in df.columns]
        if any("symptom" in c for c in cols_l) and any(k in cols_l for k in ("weight", "severity", "value")):
            # parse weights
            sym_col = next((c for c in df.columns if "symptom" in c.lower()), None)
            wt_col = next((c for c in df.columns if c and c.lower() in ("weight", "severity", "value")), None)
            if sym_col:
                for _, row in df.iterrows():
                    s = row.get(sym_col)
                    w = row.get(wt_col) if wt_col else None
                    if s:
                        try:
                            wf = float(w) if w not in (None, "", "nan") else 1.0
                        except Exception:
                            wf = 1.0
                        symptom_weights[normalize_text(s)] = wf

    # Normalize symptom_weights to 0..1 (divide by max)
    if symptom_weights:
        max_w = max(symptom_weights.values())
        if max_w > 0:
            for k in list(symptom_weights.keys()):
                symptom_weights[k] = symptom_weights[k] / max_w

    # Ensure disease_to_symptoms keys have unique normalized symptom list
    disease_to_symptoms_clean = {}
    for d, syms in disease_to_symptoms.items():
        normed = list(dict.fromkeys([normalize_text(s) for s in syms if s and str(s).strip()]))
        disease_to_symptoms_clean[d] = normed

    # convert diets_map lists -> already handled by parse_possible_list_field
    # Ensure all meta maps keys normalized
    def normalize_map(m):
        out = {}
        for k, v in m.items():
            out[normalize_text(k)] = v
        return out

    return (disease_to_symptoms_clean,
            normalize_map(description_map),
            normalize_map(meds_map),
            normalize_map(diets_map),
            normalize_map(workout_map),
            normalize_map(precautions_map),
            symptom_weights)

# ---------- Matching & scoring ----------
def match_and_score(user_symptoms: List[str], disease_to_symptoms: Dict[str, List[str]],
                    symptom_weights: Dict[str, float], user_guess: str = None, guess_conf: int = 0):
    """
    Returns ranked list of dicts: disease, score_percent, matched_symptoms, raw_score
    """
    user_norm = [normalize_text(s) for s in user_symptoms]
    max_possible = sum(1.0 for _ in user_norm) or 1.0  # simple normalization
    results = []
    for disease, dsymps in disease_to_symptoms.items():
        matched = []
        raw = 0.0
        for us in user_norm:
            best_score = 0
            best_sym = None
            for ds in dsymps:
                sc = fuzzy_ratio(us, ds)
                if sc > best_score:
                    best_score = sc
                    best_sym = ds
            # convert best_score 0..100 to 0..1
            best_score_f = best_score / 100.0
            if best_score_f >= 0.30:  # somewhat permissive threshold
                weight = symptom_weights.get(us, symptom_weights.get(best_sym, 1.0))
                contrib = best_score_f * weight
                raw += contrib
                matched.append((best_sym or us, int(best_score_f * 100)))
        # score percent
        score_percent = int((raw / max_possible) * 100)
        # apply guess boost
        guess_boost_info = None
        if user_guess:
            gscore = fuzzy_ratio(user_guess, disease) / 100.0
            if gscore >= 0.80:
                boost = int(score_percent * (min(guess_conf/100.0, 1.0) * 0.20))
                score_percent = min(100, score_percent + boost)
                guess_boost_info = {"gscore": round(gscore, 2), "boost": boost}
        results.append({
            "disease": disease,
            "raw": raw,
            "score_percent": score_percent,
            "matched": matched,
            "guess_boost": guess_boost_info
        })
    results.sort(key=lambda x: (-x["score_percent"], -x["raw"]))
    return results

# ---------- DB logging ----------
def init_db(path: str):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            query_text TEXT,
            parsed_symptoms TEXT,
            top_results TEXT,
            triage TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_query(db: str, user_id: int, username: str, query_text: str, parsed_symptoms, top_results, triage: str):
    ts = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("""
        INSERT INTO queries (user_id, username, query_text, parsed_symptoms, top_results, triage, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, username or "", query_text[:2000], json.dumps(parsed_symptoms, ensure_ascii=False),
          json.dumps(top_results, ensure_ascii=False), triage, ts))
    conn.commit()
    conn.close()

# ---------- PDF generation ----------
def generate_pdf_report(diagnosis: Dict, metadata: Dict, filename: str):
    """
    diagnosis: dictionary with keys:
      'user_symptoms', 'top_results' (list), 'triage', 'user_guess', 'user_guess_conf'
    metadata: dictionary with maps description, meds, diets, workout, precautions
    """
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Symptom Checker — Diagnostic Report (Prototype)", styles["Title"]))
    story.append(Spacer(1, 12))

    meta_lines = []
    meta_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    if diagnosis.get("user_guess"):
        meta_lines.append(f"User guess: {diagnosis.get('user_guess')} (confidence: {diagnosis.get('user_guess_conf',0)}%)")
    meta_lines.append(f"Triage suggestion: {diagnosis.get('triage')}")
    for ml in meta_lines:
        story.append(Paragraph(ml, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Reported symptoms:", styles["Heading3"]))
    for s in diagnosis.get("user_symptoms", []):
        story.append(Paragraph(f"- {s}", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Top possible conditions:", styles["Heading3"]))
    # list top results
    items = []
    for r in diagnosis.get("top_results", [])[:10]:
        title = f"{r['disease'].title()} — Confidence: {r['score_percent']}%"
        desc = []
        if r.get("matched"):
            desc.append("Matched symptoms: " + ", ".join([f"{m[0]}({m[1]}%)" for m in r["matched"]]))
        # append meta info if available
        dkey = r["disease"]
        if metadata["description"].get(dkey):
            desc.append("Description excerpt: " + (metadata["description"][dkey][:200] + ("..." if len(metadata["description"][dkey])>200 else "")))
        # Compose paragraph
        text = "<b>" + title + "</b><br/>" + "<br/>".join(desc)
        items.append(ListItem(Paragraph(text, styles["Normal"])))
    if items:
        story.append(ListFlowable(items, bulletType="1"))
    else:
        story.append(Paragraph("No matches found.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Detailed section for top suggestion
    top = diagnosis.get("top_results", [None])[0]
    if top:
        dkey = top["disease"]
        story.append(Paragraph("Top suggestion — Details", styles["Heading3"]))
        story.append(Paragraph(top["disease"].title() + f" — Confidence: {top['score_percent']}%", styles["Heading4"]))
        if metadata["description"].get(dkey):
            story.append(Paragraph("Description:", styles["Heading5"]))
            story.append(Paragraph(metadata["description"][dkey], styles["Normal"]))
        if metadata["meds"].get(dkey):
            story.append(Paragraph("Medications / Recommendations:", styles["Heading5"]))
            story.append(Paragraph(str(metadata["meds"][dkey]), styles["Normal"]))
        if metadata["diets"].get(dkey):
            story.append(Paragraph("Diet suggestions:", styles["Heading5"]))
            diet_entries = metadata["diets"][dkey]
            if isinstance(diet_entries, (list, tuple)):
                for di in diet_entries:
                    story.append(Paragraph(f"- {di}", styles["Normal"]))
            else:
                story.append(Paragraph(str(diet_entries), styles["Normal"]))
        if metadata["workout"].get(dkey):
            story.append(Paragraph("Workout suggestions:", styles["Heading5"]))
            story.append(Paragraph(str(metadata["workout"][dkey]), styles["Normal"]))
        if metadata["precautions"].get(dkey):
            story.append(Paragraph("Precautions:", styles["Heading5"]))
            precs = metadata["precautions"][dkey]
            if isinstance(precs, (list, tuple)):
                for pr in precs:
                    story.append(Paragraph(f"- {pr}", styles["Normal"]))
            else:
                story.append(Paragraph(str(precs), styles["Normal"]))

    # Footer/disclaimer
    story.append(Spacer(1, 16))
    story.append(Paragraph("Disclaimer: This report is educational only and not a medical diagnosis. Consult a qualified healthcare professional for diagnosis and treatment.", styles["Italic"]))

    doc.build(story)

# ---------- Telegram Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hello — I'm a *symptom helper (prototype)*.\n\n"
        "Send me a list of your symptoms (comma-separated), e.g.:\n"
        "`fever, cough, sore throat`\n\n"
        "You can also send a guess first like: `I think I have malaria` and then list symptoms.\n\n"
        "Use /help for more info. ⚠️ This is educational only."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send symptoms like `fever, cough` and I'll return possible causes (prototype).\n"
        "After results appear you'll be offered a 'Get PDF report' button to receive a PDF of the diagnosis.",
        parse_mode=ParseMode.MARKDOWN
    )

def parse_symptoms_from_text(text: str) -> List[str]:
    # If user wrote a guess sentence like "I think I have malaria", we will still extract words as symptoms if any listed.
    pieces = re.split(r",|;|\band\b|\n", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in pieces if p.strip()]
    return parts

def detect_guess_from_text(text: str) -> Tuple[str,int]:
    """
    If user includes "I think I have X" or "I have X" heuristically capture guess.
    Returns (guess, confidence_estimate)
    """
    t = text.lower()
    # naive patterns
    m = re.search(r"(i think i have|i have|maybe i have|i may have)\s+([a-zA-Z0-9 \-]+)", t)
    if m:
        guess = m.group(2).strip()
        # crude confidence
        if "think" in m.group(1) or "maybe" in m.group(1):
            conf = 50
        else:
            conf = 80
        return (guess, conf)
    return ("", 0)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    user = update.message.from_user
    user_id = user.id
    username = user.username
    # detect guess
    guess, guess_conf = detect_guess_from_text(text)
    # If text contains commas, treat as list of symptoms
    # Save conversation state
    # Parse symptoms from text
    symptoms = parse_symptoms_from_text(text)
    # If guess detected and user also typed symptoms, we will use both
    # If message is a simple single symptom words, still works
    if not symptoms:
        await update.message.reply_text("❌ I didn't find symptoms. Send symptoms like: `fever, cough` or type 'I have malaria, fever, chills'.")
        return

    # Build analysis
    disease_kb = context.bot_data.get("disease_to_symptoms", {})
    symptom_weights = context.bot_data.get("symptom_weights", {})
    matched = match_and_score(symptoms, disease_kb, symptom_weights, user_guess=guess or None, guess_conf=guess_conf)

    # triage check
    joined_norm = " ".join([normalize_text(s) for s in symptoms])
    triage = "non-urgent"
    if any(ek in joined_norm for ek in EMERGENCY_KEYWORDS):
        triage = "emergency"

    # prepare top list for message & keyboard
    top = matched[:TOP_N]
    lines = []
    lines.append("*Prototype Symptom Analysis* (educational only)\n")
    lines.append(f"*Parsed symptoms:* {', '.join(symptoms)}\n")
    lines.append("*Top possible conditions:*")
    if not top or top[0]["score_percent"] == 0:
        lines.append("No good matches found in the prototype knowledge base. Try adding more specific symptoms (e.g., 'fever 39C', 'productive cough').")
    else:
        for r in top:
            lines.append(f"*{r['disease'].title()}* — Confidence: {r['score_percent']}%")

    lines.append("")  # blank
    if triage == "emergency":
        lines.append("⚠️ *Possible emergency.* Seek immediate medical help or call emergency services.")
    else:
        lines.append("✅ *Non-urgent.* Consider self-care or schedule a primary care visit if symptoms persist or worsen.")

    if top and top[0]["score_percent"] < MIN_CONFIDENCE:
        lines.append("\n⚠️ *Low confidence.* Results are uncertain — consider giving more symptoms or using validated clinical API in production.")

    # keyboard: Refine / PDF / Done
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Refine results", callback_data="refine"),
        InlineKeyboardButton("Get PDF report", callback_data="get_pdf"),
        InlineKeyboardButton("I'm done", callback_data="done")
    ]])

    # store last query in chat_data for refine flow and for generating PDF
    context.chat_data["last_query"] = {
        "text": text,
        "symptoms": symptoms,
        "top_results": top,
        "triage": triage,
        "guess": guess,
        "guess_conf": guess_conf
    }

    try:
        log_query(LOG_DB, user_id, username, text, symptoms, top, triage)
    except Exception:
        logger.exception("Couldn't log query")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN, reply_markup=kb)

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    chat_id = query.message.chat_id
    user = query.from_user

    if data == "refine":
        await query.edit_message_text("Okay — to refine, tell me one more specific symptom, or tell me duration (e.g., '2 days') and any red flags (fever 39C, chest pain).")
        context.chat_data["expecting_refine"] = True
        return
    if data == "done":
        await query.edit_message_text("Thanks — remember: this is educational only. Consult a clinician for diagnosis.")
        return
    if data == "get_pdf":
        last = context.chat_data.get("last_query")
        if not last:
            await query.edit_message_text("No recent analysis in this chat to generate a PDF from. Send your symptoms first.")
            return
        # Build metadata maps
        metadata = {
            "description": context.bot_data.get("description_map", {}),
            "meds": context.bot_data.get("meds_map", {}),
            "diets": context.bot_data.get("diets_map", {}),
            "workout": context.bot_data.get("workout_map", {}),
            "precautions": context.bot_data.get("precautions_map", {})
        }
        # Create diagnosis payload
        diagnosis = {
            "user_symptoms": last["symptoms"],
            "top_results": last["top_results"],
            "triage": last["triage"],
            "user_guess": last.get("guess") or "",
            "user_guess_conf": last.get("guess_conf") or 0
        }
        # Generate PDF to temp file
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tf_path = tf.name
        tf.close()
        try:
            generate_pdf_report(diagnosis, metadata, tf_path)
            # Send PDF file
            with open(tf_path, "rb") as fh:
                # use InputFile to set a nice filename
                await context.bot.send_document(chat_id=chat_id, document=InputFile(fh, filename="diagnosis_report.pdf"), caption="Here is your diagnosis PDF (educational only).")
        except Exception as e:
            logger.exception("PDF generation or send failed: %s", e)
            await query.message.reply_text("Sorry, could not generate PDF at this time.")
        finally:
            try:
                os.unlink(tf_path)
            except Exception:
                pass
        return

async def handle_followup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If user was expected to refine, re-run analysis combining previous and new symptom
    if context.chat_data.get("expecting_refine"):
        context.chat_data["expecting_refine"] = False
        prev = context.chat_data.get("last_query") or {}
        prev_text = prev.get("text", "")
        new_text = update.message.text or ""
        combined = (prev_text + ", " + new_text).strip(", ")
        # Re-run analysis
        symptoms = parse_symptoms_from_text(combined)
        disease_kb = context.bot_data.get("disease_to_symptoms", {})
        symptom_weights = context.bot_data.get("symptom_weights", {})
        matched = match_and_score(symptoms, disease_kb, symptom_weights, user_guess=prev.get("guess"), guess_conf=prev.get("guess_conf", 0))
        triage = "non-urgent"
        if any(ek in " ".join([normalize_text(s) for s in symptoms]) for ek in EMERGENCY_KEYWORDS):
            triage = "emergency"
        top = matched[:TOP_N]
        lines = ["*Refined analysis*\n"]
        if top:
            for r in top:
                lines.append(f"*{r['disease'].title()}* — Confidence: {r['score_percent']}%")
        else:
            lines.append("No good matches found after refinement.")
        if triage == "emergency":
            lines.append("\n⚠️ *Possible emergency.* Seek immediate medical help.")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        # update last_query
        context.chat_data["last_query"] = {
            "text": combined,
            "symptoms": symptoms,
            "top_results": top,
            "triage": triage,
            "guess": prev.get("guess"),
            "guess_conf": prev.get("guess_conf", 0)
        }
    else:
        # normal message flow
        await handle_message(update, context)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Update caused error: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text("⚠️ Sorry, something went wrong in the bot. Try again later.")
    except Exception:
        pass

# ---------- Main ----------
def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN environment variable before running.")
    logger.info("Loading KB files...")
    (disease_to_symptoms,
     description_map,
     meds_map,
     diets_map,
     workout_map,
     precautions_map,
     symptom_weights) = load_kb_files(KB_PATHS)

    logger.info("Loaded %d diseases and %d symptom weights", len(disease_to_symptoms), len(symptom_weights))
    # Normalize diets_map: ensure lists or strings
    for k, v in diets_map.items():
        if isinstance(v, list):
            diets_map[k] = v
        else:
            diets_map[k] = parse_possible_list_field(v)

    # Store into bot_data for handlers
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.bot_data["disease_to_symptoms"] = disease_to_symptoms
    app.bot_data["description_map"] = description_map
    app.bot_data["meds_map"] = meds_map
    app.bot_data["diets_map"] = diets_map
    app.bot_data["workout_map"] = workout_map
    app.bot_data["precautions_map"] = precautions_map
    app.bot_data["symptom_weights"] = symptom_weights

    # Init DB
    init_db(LOG_DB)

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CallbackQueryHandler(callback_query_handler))
    # message handler: when expecting refine, handle_followup else handle_message
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_followup))
    app.add_error_handler(error_handler)

    logger.info("Bot starting (polling).")
    app.run_polling()

if __name__ == "__main__":
    main()
