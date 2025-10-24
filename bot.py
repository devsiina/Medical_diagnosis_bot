import os
import csv
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# --- CONFIG ---
TOKEN = os.getenv("TELEGRAM_TOKEN")  # set in Render environment or .env
DATASET_DIR = "dataset"
# ---------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Data ---
def read_csv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def load_datasets():
    datasets = {}
    datasets["description"] = read_csv(os.path.join(DATASET_DIR, "description.csv"))
    datasets["medications"] = read_csv(os.path.join(DATASET_DIR, "medications.csv"))
    datasets["diets"] = read_csv(os.path.join(DATASET_DIR, "diets.csv"))
    datasets["workouts"] = read_csv(os.path.join(DATASET_DIR, "workout_df.csv"))
    datasets["precautions"] = read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
    datasets["severity"] = read_csv(os.path.join(DATASET_DIR, "Symptom-severity.csv"))
    datasets["symptoms"] = read_csv(os.path.join(DATASET_DIR, "symptoms_df.csv"))
    return datasets

KB = load_datasets()

# --- Bot Logic ---
def get_disease_info(disease_name):
    disease_name = disease_name.lower()
    info = {"description": "", "medications": [], "diets": [], "workouts": [], "precautions": []}

    for d in KB["description"]:
        if d["Disease"].lower() == disease_name:
            info["description"] = d.get("Description", "")
    for d in KB["medications"]:
        if d["Disease"].lower() == disease_name:
            info["medications"].append(d.get("medication", ""))
    for d in KB["diets"]:
        if d["Disease"].lower() == disease_name:
            row_items = [v for k, v in d.items() if v and k != "Disease"]
            info["diets"].extend(row_items)
    for d in KB["workouts"]:
        if d["Disease"].lower() == disease_name:
            row_items = [v for k, v in d.items() if v and k != "Disease"]
            info["workouts"].extend(row_items)
    for d in KB["precautions"]:
        if d["Disease"].lower() == disease_name:
            row_items = [v for k, v in d.items() if v and k != "Disease"]
            info["precautions"].extend(row_items)
    return info


def guess_disease(symptoms):
    matches = {}
    for row in KB["symptoms"]:
        disease = row["Disease"]
        disease_symptoms = [v.lower() for k, v in row.items() if v and k != "Disease"]
        score = sum(1 for s in symptoms if s in disease_symptoms)
        if score > 0:
            matches[disease] = score
    if not matches:
        return None
    # pick the disease with highest symptom match
    return max(matches, key=matches.get)


# --- Handlers ---
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "👋 Hello! I'm your health assistant bot.\n"
        "Please tell me your symptoms (comma-separated), e.g. `fever, headache, sore throat`.\n\n"
        "If you already have a guess, type it like this:\n`I think I have malaria`"
    )


def handle_message(update: Update, context: CallbackContext):
    text = update.message.text.lower().strip()
    if text.startswith("i think i have"):
        disease_guess = text.replace("i think i have", "").strip()
        update.message.reply_text(f"Okay, you think you have *{disease_guess.title()}*.\nHow sure are you (0-100%)?")
        context.user_data["disease_guess"] = disease_guess
        context.user_data["awaiting_confidence"] = True
        return

    if context.user_data.get("awaiting_confidence"):
        try:
            confidence = int(text.replace("%", "").strip())
        except ValueError:
            update.message.reply_text("Please enter a number between 0 and 100.")
            return
        context.user_data["confidence"] = confidence
        context.user_data["awaiting_confidence"] = False
        update.message.reply_text("Got it! Now please tell me your symptoms (comma-separated).")
        context.user_data["awaiting_symptoms"] = True
        return

    if context.user_data.get("awaiting_symptoms"):
        symptoms = [s.strip().lower() for s in text.split(",")]
        context.user_data["symptoms"] = symptoms
        context.user_data["awaiting_symptoms"] = False
        update.message.reply_text("Thanks! Please rate the severity (1-5) of each symptom, separated by commas.")
        update.message.reply_text(f"Example for {len(symptoms)} symptoms: `3,4,2,...`")
        context.user_data["awaiting_severity"] = True
        return

    if context.user_data.get("awaiting_severity"):
        try:
            severities = [int(x.strip()) for x in text.split(",")]
        except ValueError:
            update.message.reply_text("Please enter numbers like `3,4,2`.")
            return
        symptoms = context.user_data.get("symptoms", [])
        if len(severities) != len(symptoms):
            update.message.reply_text(f"You entered {len(severities)} severities for {len(symptoms)} symptoms. Try again.")
            return

        disease_guess = context.user_data.get("disease_guess")
        guessed_disease = guess_disease(symptoms)

        final_disease = disease_guess if disease_guess else guessed_disease
        if not final_disease:
            update.message.reply_text("Sorry, I couldn't determine a likely condition from those symptoms.")
            return

        info = get_disease_info(final_disease)
        response = f"🩺 *Diagnosis (Prototype)*\n\nLikely disease: *{final_disease.title()}*\n\n"
        response += f"📖 Description:\n{info['description']}\n\n"
        if info["medications"]:
            response += f"💊 Medications:\n- " + "\n- ".join(info["medications"]) + "\n\n"
        if info["diets"]:
            response += f"🥗 Diets:\n- " + "\n- ".join(info["diets"]) + "\n\n"
        if info["workouts"]:
            response += f"🏃 Workouts:\n- " + "\n- ".join(info["workouts"]) + "\n\n"
        if info["precautions"]:
            response += f"⚠️ Precautions:\n- " + "\n- ".join(info["precautions"]) + "\n\n"
        response += "_Disclaimer: This is an informational prototype only. Consult a qualified healthcare professional for actual diagnosis._"

        update.message.reply_text(response, parse_mode="Markdown")
        context.user_data.clear()
        return

    # Default case: treat as symptom list
    symptoms = [s.strip().lower() for s in text.split(",")]
    guessed_disease = guess_disease(symptoms)
    if not guessed_disease:
        update.message.reply_text("Sorry, I couldn't find a likely disease from those symptoms.")
        return
    info = get_disease_info(guessed_disease)
    response = f"🩺 *Diagnosis (Prototype)*\n\nLikely disease: *{guessed_disease.title()}*\n\n"
    response += f"📖 Description:\n{info['description']}\n\n"
    update.message.reply_text(response, parse_mode="Markdown")


# --- Main ---
def main():
    if not TOKEN:
        raise RuntimeError("❌ TELEGRAM_TOKEN not set in environment!")

    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    logger.info("🤖 Bot is running...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
