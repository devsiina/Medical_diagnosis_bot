"""
Telegram Illness Prediction Bot (Render Deployment-Ready)
Educational Prototype — not a medical diagnostic tool.
"""

import os
import csv
import ast
import logging
from datetime import datetime, timezone
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ============= CONFIG =============
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Set this in Render Environment
KB_PATHS = [
    "dataset/Training.csv",
    "dataset/symptoms_df.csv",
    "dataset/description.csv",
    "dataset/medications.csv",
    "dataset/diets.csv",
    "dataset/workout_df.csv",
    "dataset/precautions_df.csv",
    "dataset/Symptom-severity.csv",
]
# =================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= DATA LOADING =============

def load_kb_files(kb_paths):
    print("[INFO] Loading KB files...")

    def read_csv(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                if not isinstance(row, dict):
                    try:
                        row = dict(zip(reader.fieldnames, list(row)))
                    except Exception:
                        continue
                data.append(row)
            return data

    (
        training_path,
        symptoms_path,
        desc_path,
        meds_path,
        diets_path,
        workouts_path,
        precautions_path,
        severity_path,
    ) = kb_paths

    training_data = read_csv(training_path)
    symptoms_df = read_csv(symptoms_path)
    desc_df = read_csv(desc_path)
    meds_df = read_csv(meds_path)
    diets_df = read_csv(diets_path)
    workouts_df = read_csv(workouts_path)
    precautions_df = read_csv(precautions_path)
    severity_df = read_csv(severity_path)

    # --- Normalize diets if stored as list strings ---
    for row in diets_df:
        for k, v in list(row.items()):
            if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                try:
                    items = ast.literal_eval(v)
                    if isinstance(items, list):
                        row[k] = ", ".join(items)
                except Exception:
                    pass

    print("[INFO] All KB files loaded successfully.")
    return (
        training_data,
        symptoms_df,
        desc_df,
        meds_df,
        diets_df,
        workouts_df,
        precautions_df,
        severity_df,
    )

# Load knowledge base on startup
(
    TRAINING_DATA,
    SYMPTOM_DF,
    DESC_DF,
    MEDS_DF,
    DIETS_DF,
    WORKOUT_DF,
    PRECAUTIONS_DF,
    SEVERITY_DF,
) = load_kb_files(KB_PATHS)

# ============= HELPERS =============

def find_disease_by_symptoms(symptoms):
    scores = {}
    for row in TRAINING_DATA:
        disease = row.get("Disease") or row.get("disease")
        if not disease:
            continue
        match_score = 0
        for s in symptoms:
            if s in row and row[s] == "1":
                match_score += 1
        if match_score > 0:
            scores[disease] = scores.get(disease, 0) + match_score
    if not scores:
        return None
    return max(scores, key=scores.get)

def get_description(disease):
    for row in DESC_DF:
        if row.get("Disease") == disease:
            return row.get("Description")
    return "No description available."

def get_medications(disease):
    return [r.get("medication") for r in MEDS_DF if r.get("Disease") == disease]

def get_diets(disease):
    return [v for r in DIETS_DF for v in r.values() if disease.lower() in str(v).lower()]

def get_workouts(disease):
    return [r.get("Workout") for r in WORKOUT_DF if r.get("Disease") == disease]

def get_precautions(disease):
    recs = []
    for r in PRECAUTIONS_DF:
        if r.get("Disease") == disease:
            for k, v in r.items():
                if k.startswith("Precaution") and v:
                    recs.append(v)
    return recs

def generate_pdf(disease, desc, meds, diets, workouts, precs, user_symptoms):
    filename = f"Diagnosis_{disease.replace(' ', '_')}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, f"Diagnosis Report: {disease}")
    c.setFont("Helvetica", 12)
    c.drawString(50, 780, f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    c.drawString(50, 760, f"Symptoms Provided: {', '.join(user_symptoms)}")

    text = c.beginText(50, 730)
    text.setFont("Helvetica", 12)
    text.textLines([
        f"\nDescription: {desc}",
        f"\nMedications: {', '.join(meds) if meds else 'N/A'}",
        f"\nDiet: {', '.join(diets) if diets else 'N/A'}",
        f"\nWorkouts: {', '.join(workouts) if workouts else 'N/A'}",
        f"\nPrecautions: {', '.join(precs) if precs else 'N/A'}",
        "\n\nDisclaimer: This report is informational only and not medical advice."
    ])
    c.drawText(text)
    c.save()
    return filename

# ============= BOT LOGIC =============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 Hello! I’m your Symptom Checker Bot (educational use only).\n\n"
        "Do you have a disease in mind? (yes/no)"
    )
    context.user_data.clear()
    await update.message.reply_text(msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    data = context.user_data

    if "step" not in data:
        if text in ["yes", "y"]:
            data["step"] = "ask_disease_guess"
            await update.message.reply_text("What disease do you think you have?")
        elif text in ["no", "n"]:
            data["step"] = "ask_symptoms"
            await update.message.reply_text("Please list your symptoms (comma-separated):")
        else:
            await update.message.reply_text("Please reply with 'yes' or 'no'.")
        return

    # Flow control
    if data["step"] == "ask_disease_guess":
        data["disease_guess"] = text
        data["step"] = "ask_confidence"
        await update.message.reply_text("How sure are you (0–100%)?")
        return

    if data["step"] == "ask_confidence":
        data["confidence"] = text
        data["step"] = "ask_symptoms"
        await update.message.reply_text("Now, please list your symptoms (comma-separated):")
        return

    if data["step"] == "ask_symptoms":
        symptoms = [s.strip().lower() for s in text.split(",") if s.strip()]
        data["symptoms"] = symptoms
        disease = find_disease_by_symptoms(symptoms)
        if not disease:
            await update.message.reply_text("Sorry, I couldn't identify a matching disease.")
            return

        desc = get_description(disease)
        meds = get_medications(disease)
        diets = get_diets(disease)
        workouts = get_workouts(disease)
        precs = get_precautions(disease)

        result = (
            f"🩺 *Predicted Disease:* {disease}\n\n"
            f"🧾 *Description:* {desc}\n\n"
            f"💊 *Medications:* {', '.join(meds) if meds else 'N/A'}\n"
            f"🥗 *Diet:* {', '.join(diets) if diets else 'N/A'}\n"
            f"🏃 *Workouts:* {', '.join(workouts) if workouts else 'N/A'}\n"
            f"⚠️ *Precautions:* {', '.join(precs) if precs else 'N/A'}\n\n"
            f"Would you like me to generate a PDF report? (yes/no)"
        )

        data.update({
            "predicted_disease": disease,
            "desc": desc,
            "meds": meds,
            "diets": diets,
            "workouts": workouts,
            "precs": precs,
        })
        data["step"] = "ask_pdf"
        await update.message.reply_text(result, parse_mode=ParseMode.MARKDOWN)
        return

    if data["step"] == "ask_pdf":
        if text in ["yes", "y"]:
            pdf_path = generate_pdf(
                data["predicted_disease"], data["desc"],
                data["meds"], data["diets"],
                data["workouts"], data["precs"],
                data.get("symptoms", [])
            )
            await update.message.reply_document(InputFile(pdf_path))
            os.remove(pdf_path)
        else:
            await update.message.reply_text("Okay! No PDF generated.")
        data.clear()
        await update.message.reply_text("Would you like to check another case? (yes/no)")

# ============= MAIN =============

def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN in Render environment variables.")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
