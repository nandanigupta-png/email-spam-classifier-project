"""Step 4: Bar chart of spam vs legitimate messages (Appendix C.2)."""
import os
import matplotlib

matplotlib.use("Agg")  # saves the image to a file; works without a display
import matplotlib.pyplot as plt

from prepare_data import load_and_prepare


def main(data_path="data/spam_dataset.csv", out_path="outputs/message_counts.png"):
    df = load_and_prepare(data_path)

    df["label"].value_counts().plot(kind="bar")
    plt.title("Spam and Legitimate Messages")
    plt.xlabel("Message Type")
    plt.ylabel("Number of Messages")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print("Chart saved to", out_path)
    # plt.show()  # un-comment this line to open the chart in a window


if __name__ == "__main__":
    main()
