import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VALIDATION_PATH = os.path.join(BASE_DIR, "data", "processed", "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(MODEL_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["customer_text"].fillna("")
y_train = train_df["final_intent"]

X_validation = validation_df["customer_text"].fillna("")
y_validation = validation_df["final_intent"]

X_test = test_df["customer_text"].fillna("")
y_test = test_df["final_intent"]

best_model = None
best_score = -1
best_params = None

for ngram_range in [(1, 1), (1, 2)]:
    for min_df in [1, 2]:
        for C in [0.5, 1.0, 2.0, 4.0]:
            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=True
                )),
                ("classifier", LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced"
                ))
            ])

            model.fit(X_train, y_train)

            validation_predictions = model.predict(X_validation)
            validation_score = f1_score(
                y_validation,
                validation_predictions,
                average="macro",
                zero_division=0
            )

            if validation_score > best_score:
                best_score = validation_score
                best_model = model
                best_params = {
                    "ngram_range": ngram_range,
                    "min_df": min_df,
                    "C": C
                }

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

report = classification_report(
    y_test,
    test_predictions,
    zero_division=0
)

model_path = os.path.join(MODEL_DIR, "intent_classifier.joblib")
results_path = os.path.join(
    RESULTS_DIR,
    "tfidf_logistic_regression_results.txt"
)

joblib.dump(best_model, model_path)

with open(results_path, "w", encoding="utf-8") as f:
    f.write("TF-IDF + Logistic Regression Baseline\n")
    f.write("=====================================\n\n")
    f.write(f"Best Parameters: {best_params}\n")
    f.write(f"Validation Macro F1: {best_score:.4f}\n\n")
    f.write(f"Test Examples: {len(test_df)}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Macro F1: {test_macro_f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print()
print("TF-IDF + Logistic Regression Baseline")
print("=====================================")
print()
print(f"Best Parameters: {best_params}")
print(f"Validation Macro F1: {best_score:.4f}")
print()
print(f"Test Examples: {len(test_df)}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print()
print("Classification Report:")
print(report)
print(f"Saved model to: {model_path}")
print(f"Saved results to: {results_path}")