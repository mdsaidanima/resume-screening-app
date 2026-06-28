"""
Model Retraining Script
This script retrains the model using feedback data collected from the app.
Run this periodically to improve model performance.
"""

import csv
import json
import pickle
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FEEDBACK_FILE = BASE_DIR / "feedback" / "feedback_log.json"
DATASET_FILE = ROOT_DIR / "data" / "resume_training_data.csv"
MODEL_PATHS = {
    "clf": BASE_DIR / "clf.pkl",
    "tfidf": BASE_DIR / "tfidf.pkl",
    "encoder": BASE_DIR / "encoder.pkl",
}
KNOWN_CATEGORIES = {"Data Science", "Java Developer", "Web Development", "Database Administrator", "DevOps"}


def cleanResume(txt):
    """Clean resume text"""
    if not txt:
        return ""
    special_chars = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(special_chars), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText.strip()


def load_feedback_data():
    """Load feedback data from feedback log"""
    if not FEEDBACK_FILE.exists():
        print("⚠️  No feedback data found. Create some feedback first!")
        print("   1. Upload resumes in your Streamlit app")
        print("   2. Submit feedback on predictions")
        print("   3. Then run this script again")
        return None
    
    try:
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            feedback_list = json.load(f)
    except Exception as e:
        print(f"❌ Error reading feedback file: {str(e)}")
        return None
    
    if len(feedback_list) < 1:
        print(f"⚠️  Only {len(feedback_list)} feedback records found. At least 1 needed for retraining.")
        return None
    
    print(f"✅ Loaded {len(feedback_list)} feedback records")
    return feedback_list


def load_training_dataset(data_file=None):
    """Load additional resume examples from a CSV file in the data folder."""
    path = Path(data_file) if data_file else DATASET_FILE
    if not path.exists():
        print(f"ℹ️  No dataset file found at {path}")
        return []

    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                text = (row.get('text') or row.get('resume_text') or '').strip()
                category = (row.get('category') or row.get('label') or '').strip()
                if text and category:
                    rows.append({"text": text, "category": category})
        print(f"✅ Loaded {len(rows)} rows from {path}")
        return rows
    except Exception as e:
        print(f"❌ Error reading dataset file: {str(e)}")
        return []


def prepare_training_examples(feedback_list, extra_rows=None):
    """Build clean text and label lists from feedback plus optional dataset rows."""
    texts = []
    labels = []

    for feedback in feedback_list or []:
        actual_category = feedback.get('actual')
        resume_text = feedback.get('resume_text') or feedback.get('text') or ''
        cleaned_text = cleanResume(resume_text)
        if actual_category and cleaned_text and actual_category in KNOWN_CATEGORIES:
            texts.append(cleaned_text)
            labels.append(actual_category)

    for row in extra_rows or []:
        cleaned_text = cleanResume(row.get('text') or row.get('resume_text') or '')
        category = row.get('category') or row.get('label') or ''
        if cleaned_text and category and category in KNOWN_CATEGORIES:
            texts.append(cleaned_text)
            labels.append(category)

    return texts, labels


def backup_models():
    """Backup existing models"""
    import shutil
    from datetime import datetime
    
    backup_dir = BASE_DIR / "model_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_prefix = backup_dir / f"backup_{timestamp}"
    
    try:
        for name, path in MODEL_PATHS.items():
            if path.exists():
                shutil.copy(path, f"{backup_prefix}_{name}.pkl")
        print(f"✅ Models backed up to {backup_dir}")
        return True
    except Exception as e:
        print(f"⚠️  Backup failed: {str(e)}")
        return False


def retrain_model(feedback_list, extra_rows=None):
    """Retrain model with feedback data and optional starter dataset rows."""
    
    # Backup existing models
    backup_models()
    
    texts, labels = prepare_training_examples(feedback_list, extra_rows=extra_rows)
    
    if len(texts) < 2:
        print("❌ Not enough valid training examples to retrain. Need at least 2.")
        return False
    
    print(f"📊 Training with {len(texts)} examples...")
    
    try:
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        vectorized_texts = tfidf.fit_transform(texts)

        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Train new SVM model
        svc_model = SVC(kernel='rbf', gamma='scale', probability=True)
        svc_model.fit(vectorized_texts, encoded_labels)
        
        # Save new models
        with open(MODEL_PATHS['clf'], 'wb') as f:
            pickle.dump(svc_model, f)
        with open(MODEL_PATHS['tfidf'], 'wb') as f:
            pickle.dump(tfidf, f)
        with open(MODEL_PATHS['encoder'], 'wb') as f:
            pickle.dump(le, f)

        print("✅ Model retrained and saved!")
        
        # Calculate and display metrics
        train_accuracy = svc_model.score(vectorized_texts, encoded_labels)
        print(f"✅ Training accuracy: {train_accuracy*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Retraining failed: {str(e)}")
        return False


def main():
    print("\n" + "="*50)
    print("🤖 MODEL RETRAINING PIPELINE")
    print("="*50 + "\n")
    
    # Load feedback data
    feedback_list = load_feedback_data()
    extra_rows = load_training_dataset()
    
    if feedback_list is None and not extra_rows:
        print("⚠️  Exiting retraining.")
        return
    
    # Show feedback statistics
    feedback_count = len(feedback_list) if feedback_list else 0
    print(f"\n📈 Feedback Statistics:")
    print(f"   Total feedback records: {feedback_count}")
    
    if feedback_list:
        correct = sum(1 for f in feedback_list if f.get('correct', False))
        print(f"   Correct predictions: {correct}/{feedback_count} ({correct/feedback_count*100:.1f}%)")
    else:
        print("   Correct predictions: 0/0 (0.0%)")
    
    # Group by actual category
    categories = {}
    if feedback_list:
        for feedback in feedback_list:
            actual = feedback.get('actual')
            if actual:
                categories[actual] = categories.get(actual, 0) + 1
    
    print(f"\n   Categories in feedback:")
    for cat, count in sorted(categories.items()):
        print(f"      • {cat}: {count}")
    
    # Ask for confirmation
    response = input("\n⚠️  Continue with retraining? (yes/no): ").strip().lower()
    
    if response == 'yes':
        if retrain_model(feedback_list, extra_rows=extra_rows):
            print("\n✅ Retraining completed successfully!")
            print("📝 Restart your Streamlit app to use the new model.\n")
        else:
            print("\n❌ Retraining failed. Old models restored from backup.\n")
    else:
        print("❌ Retraining cancelled.\n")


if __name__ == "__main__":
    main()
