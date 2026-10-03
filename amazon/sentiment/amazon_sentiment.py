import gzip
import json
import random
import re
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PODEŠAVANJA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE_PATH = (
    PROJECT_ROOT
    / "data-service"
    / "data"
    / "amazon"
    / "Electronics_5.json.gz"
)

SAMPLES_PER_CLASS = 100_000

RANDOM_SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent

SAMPLE_FILE = OUTPUT_DIR / "amazon_sentiment_sample.csv"
RESULTS_FILE = OUTPUT_DIR / "amazon_sentiment_results.txt"


# ============================================================
# ČIŠĆENJE TEKSTA
# ============================================================

def clean_text(text):
    """
    Osnovno čišćenje teksta recenzija.
    """

    if not text:
        return ""

    text = str(text)

    # Mala slova
    text = text.lower()

    # Uklanjanje HTML oznaka
    text = re.sub(r"<[^>]+>", " ", text)

    # Uklanjanje URL adresa
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Uklanjanje znakova koji nisu slova ili brojevi
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Uklanjanje višestrukih razmaka
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# PRIKUPLJANJE BALANSIRANOG UZORKA
# ============================================================

def collect_balanced_sample(file_path):
    """
    Prolazi kroz ceo Amazon dataset i formira balansirani uzorak:
    100.000 pozitivnih + 100.000 negativnih recenzija.

    Pozitivno:
        ocene 4 i 5

    Negativno:
        ocene 1 i 2

    Ocena 3 se ne koristi.
    """

    random.seed(RANDOM_SEED)

    positive_sample = []
    negative_sample = []

    positive_seen = 0
    negative_seen = 0

    total_processed = 0

    print("=" * 60)
    print("FORMIRANJE BALANSIRANOG UZORKA")
    print("=" * 60)
    print()
    print("Cilj:")
    print(
        f"  Pozitivne recenzije: "
        f"{SAMPLES_PER_CLASS:,}"
    )
    print(
        f"  Negativne recenzije: "
        f"{SAMPLES_PER_CLASS:,}"
    )
    print()

    with gzip.open(
        file_path,
        "rt",
        encoding="utf-8"
    ) as file:

        for line in file:

            try:
                review = json.loads(line)

            except json.JSONDecodeError:
                continue

            total_processed += 1

            rating = review.get("overall")
            review_text = review.get("reviewText")

            if rating is None or not review_text:
                continue

            # ------------------------------------------------
            # POZITIVNA RECENZIJA
            # ------------------------------------------------

            if rating >= 4:

                positive_seen += 1

                item = (
                    clean_text(review_text),
                    1
                )

                if len(positive_sample) < SAMPLES_PER_CLASS:

                    positive_sample.append(item)

                else:

                    random_index = random.randint(
                        0,
                        positive_seen - 1
                    )

                    if random_index < SAMPLES_PER_CLASS:

                        positive_sample[
                            random_index
                        ] = item

            # ------------------------------------------------
            # NEGATIVNA RECENZIJA
            # ------------------------------------------------

            elif rating <= 2:

                negative_seen += 1

                item = (
                    clean_text(review_text),
                    0
                )

                if len(negative_sample) < SAMPLES_PER_CLASS:

                    negative_sample.append(item)

                else:

                    random_index = random.randint(
                        0,
                        negative_seen - 1
                    )

                    if random_index < SAMPLES_PER_CLASS:

                        negative_sample[
                            random_index
                        ] = item

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if total_processed % 500_000 == 0:

                print(
                    f"Obrađeno: {total_processed:,} | "
                    f"Pozitivnih kandidata: "
                    f"{positive_seen:,} | "
                    f"Negativnih kandidata: "
                    f"{negative_seen:,}"
                )

    # Uklanjamo prazne tekstove
    positive_sample = [
        item
        for item in positive_sample
        if item[0]
    ]

    negative_sample = [
        item
        for item in negative_sample
        if item[0]
    ]

    print()
    print("Formiranje uzorka završeno.")
    print(
        f"Pozitivnih: "
        f"{len(positive_sample):,}"
    )
    print(
        f"Negativnih: "
        f"{len(negative_sample):,}"
    )

    return positive_sample, negative_sample


# ============================================================
# ČUVANJE UZORKA
# ============================================================

def save_sample(
    positive_sample,
    negative_sample
):

    data = (
        positive_sample +
        negative_sample
    )

    random.Random(
        RANDOM_SEED
    ).shuffle(data)

    with open(
        SAMPLE_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "text",
            "sentiment"
        ])

        writer.writerows(data)

    print()
    print(
        f"Uzorak sačuvan u: "
        f"{SAMPLE_FILE}"
    )


# ============================================================
# TF-IDF + LOGISTIC REGRESSION
# ============================================================

def train_model(
    positive_sample,
    negative_sample
):

    data = (
        positive_sample +
        negative_sample
    )

    random.Random(
        RANDOM_SEED
    ).shuffle(data)

    texts = [
        item[0]
        for item in data
    ]

    labels = [
        item[1]
        for item in data
    ]

    print()
    print("=" * 60)
    print("TF-IDF TRANSFORMACIJA")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        min_df=5,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)

    print(
        f"Broj recenzija: "
        f"{X.shape[0]:,}"
    )

    print(
        f"Broj TF-IDF karakteristika: "
        f"{X.shape[1]:,}"
    )

    print(
        f"Broj nula u matrici: "
        f"{X.nnz:,}"
    )

    print()
    print("=" * 60)
    print("PODELA NA TRAIN I TEST")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    print(
        f"Train skup: "
        f"{X_train.shape[0]:,} recenzija"
    )

    print(
        f"Test skup: "
        f"{X_test.shape[0]:,} recenzija"
    )

    print()
    print("=" * 60)
    print("TRENIRANJE LOGISTIC REGRESSION MODELA")
    print("=" * 60)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model je uspešno treniran.")

    # Predikcija
    y_pred = model.predict(
        X_test
    )

    # ========================================================
    # METRIKE
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Negative",
            "Positive"
        ]
    )

    print()
    print("=" * 60)
    print("REZULTATI SENTIMENT ANALIZE")
    print("=" * 60)

    print(
        f"\nAccuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score : "
        f"{f1:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(report)

    # ========================================================
    # ČUVANJE REZULTATA
    # ========================================================

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "AMAZON SENTIMENT ANALYSIS RESULTS\n"
        )

        file.write(
            "=" * 50 +
            "\n\n"
        )

        file.write(
            f"Samples per class: "
            f"{SAMPLES_PER_CLASS:,}\n"
        )

        file.write(
            f"Total samples: "
            f"{len(data):,}\n"
        )

        file.write(
            f"Train samples: "
            f"{len(X_train):,}\n"
        )

        file.write(
            f"Test samples: "
            f"{len(X_test):,}\n\n"
        )

        file.write(
            f"TF-IDF features: "
            f"{X.shape[1]:,}\n\n"
        )

        file.write(
            f"Accuracy: "
            f"{accuracy:.4f}\n"
        )

        file.write(
            f"Precision: "
            f"{precision:.4f}\n"
        )

        file.write(
            f"Recall: "
            f"{recall:.4f}\n"
        )

        file.write(
            f"F1-score: "
            f"{f1:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix:\n"
        )

        file.write(
            str(cm) +
            "\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(
            report
        )

    print()
    print(
        f"Rezultati sačuvani u: "
        f"{RESULTS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    positive_sample, negative_sample = (
        collect_balanced_sample(
            FILE_PATH
        )
    )

    save_sample(
        positive_sample,
        negative_sample
    )

    train_model(
        positive_sample,
        negative_sample
    )

    print()
    print("=" * 60)
    print("SENTIMENT EKSPERIMENT JE ZAVRŠEN.")
    print("=" * 60)
