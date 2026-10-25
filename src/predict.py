"""Step 3: Use the saved model to classify new messages (Appendix C.1)."""
import sys
import joblib

MODEL_PATH = "models/spam_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)

    # If you type a message after the command, classify only that message
    if len(sys.argv) > 1:
        messages = [" ".join(sys.argv[1:])]
    else:
        messages = [
            "You have won a free prize!",
            "Your class starts at 10 AM.",
            "Claim your special offer now!",
        ]

    results = model.predict(messages)
    for message, result in zip(messages, results):
        print(message, ":", result)


if __name__ == "__main__":
    main()
