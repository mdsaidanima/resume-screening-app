import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "notebooks" / "retrain_model.py"

spec = importlib.util.spec_from_file_location("retrain_model", MODULE_PATH)
retrain_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain_model)


class RetrainModelTests(unittest.TestCase):
    def test_prepare_training_examples_uses_resume_text_from_feedback(self):
        feedback = [
            {"actual": "Data Science", "resume_text": "Python, SQL, machine learning"},
            {"actual": "Web Development", "resume_text": "React, JavaScript, CSS"},
            {"actual": "Data Science", "resume_text": "Pandas, scikit-learn"},
            {"actual": "Unknown", "resume_text": "Should be ignored"},
        ]

        texts, labels = retrain_model.prepare_training_examples(feedback)

        self.assertEqual(labels, ["Data Science", "Web Development", "Data Science"])
        self.assertEqual(len(texts), 3)
        self.assertIn("python", texts[0].lower())


if __name__ == "__main__":
    unittest.main()
