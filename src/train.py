"""Step 2: Train and evaluate the model (Appendix B of the report)."""
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from prepare_data import load_and_prepare


def main(data_path="data/spam_dataset.csv", model_path="models/spam_model.pkl"):
    df = load_and_prepare(data_path)

    X = df["message"]
    y = df["label"]

    # 80% training data, 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TF-IDF turns text into numbers, LinearSVC (an SVM) classifies them
    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ---- Evaluation (fills Table 3.4 of the report) ----
    print("\n===== EVALUATION RESULTS =====")
    print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(y_test, predictions, pos_label="spam", zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, predictions, pos_label="spam", zero_division=0), 4))
    print("F1-score :", round(f1_score(y_test, predictions, pos_label="spam", zero_division=0), 4))

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print("Order: [spam, legitimate]")
    print(confusion_matrix(y_test, predictions, labels=["spam", "legitimate"]))

    print("\nFull report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # C.3 - Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("Model saved to", model_path)


if __name__ == "__main__":
    main()
