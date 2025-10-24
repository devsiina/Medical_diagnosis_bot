import os
import csv
from datetime import datetime, UTC
from io import BytesIO
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, filters, ContextTypes

# === Load knowledge base ===
KB_PATHS = {
    "training": "dataset/Training.csv",
    "description": "dataset/description.csv",
    "medications": "dataset/medications.csv",
    "diets": "dataset/diets.csv",
    "workout": "dataset/workout_df.csv",
    "precautions": "dataset/precautions_df.csv",
    "severity": "dataset/Symptom-severity.csv"
}

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def load_kb():
    return {
        k: load_csv(v)
        for k, v in KB_PATHS.items()
    }

KB = load_kb()

# === Conversation states ===
ASK_GUESS, ASK_CERTAINTY, ASK_SYMPTOMS, ASK_SEVERITY, DIAGNOSIS = range(5)

user_sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome! Let's figure out what might be wrong.\n"
        "Do you already have a guess about what disease you might have? (yes/no)"
    )
    return ASK_GUESS

async def ask_guess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower().strip()
    user_id = update.effective_user.id
    user_sessions[user_id] = {"guess": None, "certainty": 0, "symptoms": {}, "diagnosis": ""}

    if text.startswith("y"):
        await update.message.reply_text("What disease do you suspect?")
        return ASK_CERTAINTY
    else:
        await update.message.reply_text("Alright. Please list your symptoms, separated by commas.")
        return ASK_SYMPTOMS

async def ask_certainty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id]["guess"] = update.message.text.strip()
    await update.message.reply_text("On a scale of 1–100, how sure are you?")
    return ASK_SYMPTOMS

async def ask_symptoms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    if "certainty" not in user_sessions[user_id]:
        user_sessions[user_id]["certainty"] = 0
    symptoms = [s.strip().capitalize() for s in text.split(",")]
    user_sessions[user_id]["symptoms"] = {s: 0 for s in symptoms}
    await update.message.reply_text("Rate the severity (1–10) for each symptom one by one.")
    await update.message.reply_text(f"First: {symptoms[0]}")
    context.user_data["symptom_list"] = symptoms
    context.user_data["index"] = 0
    return ASK_SEVERITY

async def ask_severity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    severity = update.message.text.strip()
    if not severity.isdigit():
        await update.message.reply_text("Please enter a number (1–10).")
        return ASK_SEVERITY
    severity = int(severity)

    symptoms = context.user_data["symptom_list"]
    idx = context.user_data["index"]
    user_sessions[user_id]["symptoms"][symptoms[idx]] = severity

    idx += 1
    if idx < len(symptoms):
        context.user_data["index"] = idx
        await update.message.reply_text(f"Next: {symptoms[idx]}")
        return ASK_SEVERITY
    else:
        await update.message.reply_text("Analyzing your symptoms...")
        disease = infer_disease(user_sessions[user_id]["symptoms"])
        user_sessions[user_id]["diagnosis"] = disease
        await send_diagnosis(update, user_id)
        return ConversationHandler.END

def infer_disease(symptoms):
    # Simple rule-based mock-up; replace with actual model logic
    if "fever" in [s.lower() for s in symptoms]:
        return "Malaria"
    elif "cough" in [s.lower() for s in symptoms]:
        return "Flu"
    else:
        return "Unknown condition"

async def send_diagnosis(update: Update, user_id):
    diagnosis = user_sessions[user_id]["diagnosis"]

    desc = next((d["Description"] for d in KB["description"] if d["Disease"].lower() == diagnosis.lower()), "No description available.")
    meds = [d["medication"] for d in KB["medications"] if d["Disease"].lower() == diagnosis.lower()]
    diets = []
    for d in KB["diets"]:
        if d["Disease"].lower() == diagnosis.lower():
            for v in d.values():
                if v and "Diet" not in v:
                    diets.append(v)
    workouts = [d["Workout"] for d in KB["workout"] if d["Disease"].lower() == diagnosis.lower()]
    precautions = []
    for p in KB["precautions"]:
        if p["Disease"].lower() == diagnosis.lower():
            for k, v in p.items():
                if k != "Disease" and v:
                    precautions.append(v)

    message = (
        f"🩺 *Diagnosis:* {diagnosis}\n\n"
        f"📖 *Description:* {desc}\n\n"
        f"💊 *Medications:* {', '.join(meds) if meds else 'N/A'}\n"
        f"🥗 *Diet:* {', '.join(diets) if diets else 'N/A'}\n"
        f"🏋️ *Workouts:* {', '.join(workouts) if workouts else 'N/A'}\n"
        f"⚠️ *Precautions:* {', '.join(precautions) if precautions else 'N/A'}\n"
    )

    await update.message.reply_text(message, parse_mode="Markdown")
    await update.message.reply_text("Would you like a PDF summary? (yes/no)")

async def generate_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if "yes" not in update.message.text.lower():
        await update.message.reply_text("Alright! Stay healthy. 🌿")
        return ConversationHandler.END

    diagnosis = user_sessions[user_id]["diagnosis"]
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    pdf_content = f"""
    Medical Diagnosis Report
    -------------------------
    Patient ID: {user_id}
    Timestamp: {now}

    Diagnosis: {diagnosis}
    Symptoms: {user_sessions[user_id]["symptoms"]}
    """

    pdf = BytesIO()
    pdf.write(pdf_content.encode("utf-8"))
    pdf.seek(0)

    await update.message.reply_document(InputFile(pdf, filename="diagnosis_report.pdf"))
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Session cancelled. Stay safe!")
    return ConversationHandler.END

def main():
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_GUESS: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_guess)],
            ASK_CERTAINTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_certainty)],
            ASK_SYMPTOMS: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_symptoms)],
            ASK_SEVERITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_severity)],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    app.add_handler(conv)
    app.add_handler(MessageHandler(filters.Regex("(?i)^yes$"), generate_pdf))
    app.add_handler(MessageHandler(filters.Regex("(?i)^no$"), cancel))

    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
