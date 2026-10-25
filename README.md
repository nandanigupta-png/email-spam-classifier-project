# 📧 Email Spam Classifier

A beginner-friendly Machine Learning project that classifies a message as **spam** or **legitimate**.
It was built as the final project of the One-Month Artificial Intelligence Internship at Codec Technologies.

## How it works

1. **Load data** – read messages that are already labelled `spam` or `ham` (normal).
2. **Clean data** – remove empty rows, lowercase labels, rename `ham` to `legitimate`.
3. **TF-IDF** – convert the text into numbers.
4. **SVM (LinearSVC)** – learn to separate spam from legitimate messages.
5. **Evaluate** – accuracy, precision, recall, F1-score and confusion matrix.
6. **Predict** – classify any new message.

## Project structure

```
email-spam-classifier/
├── data/
│   └── spam_dataset.csv     # sample dataset (replace with a bigger one)
├── src/
│   ├── prepare_data.py      # load + clean data
│   ├── train.py             # train, evaluate, save model
│   ├── predict.py           # classify new messages
│   └── visualize.py         # bar chart of message counts
├── models/                  # trained model is saved here
├── outputs/                 # charts are saved here
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

You need Python 3.9 or newer.

```bash
# 1. (optional but recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. install the libraries
pip install -r requirements.txt
```

## Usage

Run these commands from the main project folder:

```bash
python src/train.py                       # train + evaluate + save the model
python src/predict.py                     # test on 3 built-in messages
python src/predict.py "Win a free prize"  # test on your own message
python src/visualize.py                   # saves outputs/message_counts.png
```

Example output:

```
You have won a free prize! : spam
Your class starts at 10 AM. : legitimate
```

## Use a bigger dataset (recommended)

The included `data/spam_dataset.csv` is a tiny sample (80 messages) so the project runs instantly.
Its perfect scores are **not realistic**. For meaningful results:

1. Download the **SMS Spam Collection** dataset from Kaggle or the UCI Machine Learning Repository.
2. Make sure the CSV has two columns named **`message`** and **`label`** (labels: `spam` / `ham`).
   If your columns have other names (for example `v1` and `v2`), rename them in `src/prepare_data.py`.
3. Save it as `data/spam_dataset.csv` and run `python src/train.py` again.
4. Copy the real Accuracy / Precision / Recall / F1 values into your report (Table 3.4).

## Limitations

- A spam message may be marked as legitimate, and the other way round.
- Results depend on the quality and size of the dataset.
- New kinds of spam may be missed.
- The model cannot guarantee that a message is safe. Be careful with suspicious links.

## Technologies

Python · pandas · scikit-learn · matplotlib · joblib

## Author

**Nandani Gupta** – B.Tech CSE, Central University of Jammu

## License

MIT License – see the `LICENSE` file.
