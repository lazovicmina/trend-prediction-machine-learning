import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# ============================================================
# PODEŠAVANJA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AMAZON_FILE = (
    PROJECT_ROOT
    / "data-service"
    / "data"
    / "amazon"
    / "Electronics_5.json.gz"
)

OUTPUT_DIR = Path(__file__).resolve().parent

INTERACTIONS_FILE = (
    PROJECT_ROOT
    / "amazon"
    / "collaborative_filtering"
    / "amazon_cf_interactions.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "amazon_cbf_evaluation_results.txt"
)

TOP_K = 10

N_TEST_ITEMS = 2
MIN_TOTAL_INTERACTIONS = 5

# Broj korisnika koji se koristi za evaluaciju.
MAX_EVALUATION_USERS = 5_000

# Broj kandidata koje tražimo za svakog korisnika.
NEIGHBORS_TO_SEARCH = 100

# TF-IDF podešavanja
MAX_FEATURES = 50_000
MIN_DF = 2
MAX_DF = 0.95

# Veličina batch-a tokom evaluacije
BATCH_SIZE = 50


# ============================================================
# 1. UČITAVANJE KORISNIČKIH INTERAKCIJA
# ============================================================

print("Učitavanje korisničkih interakcija...")

interactions = pd.read_csv(
    INTERACTIONS_FILE
)

interactions["timestamp"] = pd.to_datetime(
    interactions["timestamp"]
)

print(
    f"Interakcija: "
    f"{len(interactions):,}"
)

print(
    f"Korisnika: "
    f"{interactions['user_id'].nunique():,}"
)

print(
    f"Proizvoda: "
    f"{interactions['item_id'].nunique():,}"
)


# ============================================================
# 2. FORMIRANJE TRAIN / TEST PODELE
# ============================================================

print()
print("Formiranje train/test podele...")

user_histories = defaultdict(list)

for row in interactions.itertuples(index=False):

    user_histories[row.user_id].append(
        (
            row.timestamp,
            row.item_id
        )
    )


# Hronološko sortiranje interakcija
for user in user_histories:

    user_histories[user].sort(
        key=lambda x: x[0]
    )


train_users = {}
test_users = {}


for user, history in user_histories.items():

    # Uklanjamo duplikate proizvoda uz očuvanje
    # vremenskog redosleda.
    seen = set()
    unique_history = []

    for timestamp, item in history:

        if item not in seen:

            seen.add(item)

            unique_history.append(
                (
                    timestamp,
                    item
                )
            )

    # Korisnik mora imati najmanje 5 proizvoda.
    if len(unique_history) < MIN_TOTAL_INTERACTIONS:
        continue

    # Poslednja 2 proizvoda idu u test.
    train_part = unique_history[
        :-N_TEST_ITEMS
    ]

    test_part = unique_history[
        -N_TEST_ITEMS:
    ]

    train_users[user] = [
        item
        for _, item in train_part
    ]

    test_users[user] = [
        item
        for _, item in test_part
    ]


print(
    f"Korisnika pogodnih za evaluaciju: "
    f"{len(train_users):,}"
)


# ============================================================
# 3. IZBOR KORISNIKA ZA EVALUACIJU
# ============================================================

evaluation_users = sorted(
    train_users.keys(),
    key=lambda x: str(x)
)[:MAX_EVALUATION_USERS]


print(
    f"Korisnika koji će biti evaluirani: "
    f"{len(evaluation_users):,}"
)


# ============================================================
# 4. FORMIRANJE TRAIN USER-ITEM KLJUČEVA
# ============================================================
#
# Koristimo samo:
#
# user_id + item_id
#
# Ne koristimo vreme, jer nam ovde nije potrebno
# za povezivanje sa originalnim Amazon recenzijama.
#
# Na ovaj način sprečavamo korišćenje test proizvoda
# u tekstualnom profilu proizvoda.
# ============================================================

train_interaction_keys = set()

for user, items in train_users.items():

    for item in items:

        train_interaction_keys.add(
            (
                str(user),
                str(item)
            )
        )


print(
    f"Train user-item veza: "
    f"{len(train_interaction_keys):,}"
)


# ============================================================
# 5. ČITANJE TEKSTA IZ AMAZON SKUPA
# ============================================================

print()
print(
    "Čitanje tekstualnog sadržaja iz train dela..."
)

product_texts = defaultdict(list)

total_reviews = 0
train_reviews_used = 0


with gzip.open(
    AMAZON_FILE,
    "rt",
    encoding="utf-8"
) as f:

    for line in f:

        try:
            review = json.loads(line)

        except json.JSONDecodeError:
            continue

        total_reviews += 1

        reviewer_id = review.get(
            "reviewerID"
        )

        product_id = review.get(
            "asin"
        )

        if (
            reviewer_id is None
            or product_id is None
        ):
            continue

        key = (
            str(reviewer_id),
            str(product_id)
        )

        # U tekstualni opis proizvoda uključujemo
        # samo recenzije koje pripadaju train interakcijama.
        if key not in train_interaction_keys:
            continue

        summary = review.get(
            "summary",
            ""
        )

        review_text = review.get(
            "reviewText",
            ""
        )

        text = (
            f"{summary} {review_text}"
        ).strip()

        if not text:
            continue

        product_texts[
            str(product_id)
        ].append(text)

        train_reviews_used += 1

        if total_reviews % 500_000 == 0:

            print(
                f"Obrađeno recenzija: "
                f"{total_reviews:,}"
            )


print()

print(
    f"Ukupno obrađenih recenzija: "
    f"{total_reviews:,}"
)

print(
    f"Train recenzija korišćenih za CBF: "
    f"{train_reviews_used:,}"
)

print(
    f"Proizvoda sa tekstom: "
    f"{len(product_texts):,}"
)


# ============================================================
# 6. FORMIRANJE LISTE PROIZVODA
# ============================================================

products = sorted(
    product_texts.keys()
)


print(
    f"Proizvoda koji ulaze u CBF: "
    f"{len(products):,}"
)


if len(products) == 0:

    raise RuntimeError(
        "Nema proizvoda sa train tekstom. "
        "Provera train_interaction_keys nije uspela."
    )


# ============================================================
# 7. FORMIRANJE TEKSTUALNOG OPISA PROIZVODA
# ============================================================

texts = []

for product_id in products:

    combined_text = " ".join(
        product_texts[product_id]
    )

    texts.append(
        combined_text
    )


product_to_index = {
    product_id: index
    for index, product_id in enumerate(products)
}


# ============================================================
# 8. TF-IDF
# ============================================================

print()
print(
    "Formiranje TF-IDF reprezentacije..."
)


vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english"
)


tfidf_matrix = vectorizer.fit_transform(
    texts
)


print(
    f"TF-IDF matrica: "
    f"{tfidf_matrix.shape}"
)


# Normalizacija TF-IDF vektora
tfidf_matrix = normalize(
    tfidf_matrix,
    norm="l2",
    axis=1
)


# ============================================================
# 9. MODEL ZA PRONALAŽENJE SLIČNIH PROIZVODA
# ============================================================

print()
print(
    "Priprema modela za brzo pronalaženje "
    "sličnih proizvoda..."
)


nn_model = NearestNeighbors(
    n_neighbors=NEIGHBORS_TO_SEARCH,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)


nn_model.fit(
    tfidf_matrix
)


print(
    "Model za pretragu sličnih proizvoda je spreman."
)


# ============================================================
# 10. EVALUACIJA
# ============================================================

print()
print(
    "Evaluacija CBF modela..."
)

print(
    "--------------------------------------------"
)


precision_values = []
recall_values = []
hit_values = []

evaluated_users = 0

test_items_total = 0
test_items_known = 0


# ============================================================
# 11. BATCH OBRADA KORISNIKA
# ============================================================

for batch_start in range(
    0,
    len(evaluation_users),
    BATCH_SIZE
):

    batch_users = evaluation_users[
        batch_start:
        batch_start + BATCH_SIZE
    ]

    profile_rows = []
    valid_batch_users = []

    # --------------------------------------------------------
    # Formiranje korisničkih profila
    # --------------------------------------------------------

    for user in batch_users:

        train_items = train_users[user]

        item_indices = [
            product_to_index[item]
            for item in train_items
            if item in product_to_index
        ]

        if not item_indices:
            continue

        # Korisnički profil predstavlja prosečan
        # TF-IDF vektor proizvoda koje je korisnik
        # prethodno imao u train skupu.
        profile = tfidf_matrix[
            item_indices
        ].mean(axis=0)

        # Pretvaranje numpy.matrix objekta u
        # običnu sparse matricu.
        profile = csr_matrix(
            np.asarray(profile)
        )

        profile_rows.append(
            profile
        )

        valid_batch_users.append(
            user
        )

    if not profile_rows:
        continue

    user_profiles = vstack(
        profile_rows
    )

    user_profiles = normalize(
        user_profiles,
        norm="l2",
        axis=1
    )

    # --------------------------------------------------------
    # Pronalaženje najsličnijih proizvoda
    # --------------------------------------------------------

    distances, indices = nn_model.kneighbors(
        user_profiles,
        return_distance=True
    )

    # --------------------------------------------------------
    # Evaluacija korisnika
    # --------------------------------------------------------

    for row_index, user in enumerate(
        valid_batch_users
    ):

        train_items = set(
            train_users[user]
        )

        actual_items = set(
            test_users[user]
        )

        # Broj poznatih test proizvoda
        for item in actual_items:

            test_items_total += 1

            if item in product_to_index:

                test_items_known += 1

        recommended_items = []

        # Ne koristimo distance kao rezultat,
        # već samo redosled koji daje nearest-neighbors.
        for neighbor_index in indices[row_index]:

            candidate = products[
                neighbor_index
            ]

            # Ne preporučujemo proizvod koji je
            # korisnik već imao u train skupu.
            if candidate in train_items:
                continue

            if candidate not in recommended_items:

                recommended_items.append(
                    candidate
                )

            if len(recommended_items) >= TOP_K:

                break

        recommended_set = set(
            recommended_items
        )

        # ----------------------------------------------------
        # Broj pogodaka
        # ----------------------------------------------------

        hits = len(
            recommended_set.intersection(
                actual_items
            )
        )

        # ----------------------------------------------------
        # Precision@K
        # ----------------------------------------------------

        precision = (
            hits / TOP_K
        )

        # ----------------------------------------------------
        # Recall@K
        # ----------------------------------------------------

        recall = (
            hits / len(actual_items)
            if actual_items
            else 0
        )

        # ----------------------------------------------------
        # Hit Rate@K
        # ----------------------------------------------------

        hit_rate = (
            1
            if hits > 0
            else 0
        )

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        hit_values.append(
            hit_rate
        )

        evaluated_users += 1

    print(
        f"Evaluirano korisnika: "
        f"{min(batch_start + BATCH_SIZE, len(evaluation_users)):,}/"
        f"{len(evaluation_users):,}"
    )


# ============================================================
# 12. IZRAČUNAVANJE KONAČNIH METRIKA
# ============================================================

precision_at_k = (
    float(np.mean(precision_values))
    if precision_values
    else 0.0
)


recall_at_k = (
    float(np.mean(recall_values))
    if recall_values
    else 0.0
)


hit_rate_at_k = (
    float(np.mean(hit_values))
    if hit_values
    else 0.0
)


known_test_percentage = (
    test_items_known
    / test_items_total
    * 100
    if test_items_total
    else 0.0
)


# ============================================================
# 13. ISPIS REZULTATA
# ============================================================

print()

print(
    "=" * 60
)

print(
    "CONTENT-BASED FILTERING - REZULTATI"
)

print(
    "=" * 60
)


print(
    f"Evaluiranih korisnika: "
    f"{evaluated_users:,}"
)


print(
    f"TOP-K: "
    f"{TOP_K}"
)


print(
    f"Test proizvoda poznatih CBF modelu: "
    f"{known_test_percentage:.2f}%"
)


print()


print(
    f"Precision@{TOP_K}: "
    f"{precision_at_k:.4f}"
)


print(
    f"Recall@{TOP_K}: "
    f"{recall_at_k:.4f}"
)


print(
    f"Hit Rate@{TOP_K}: "
    f"{hit_rate_at_k:.4f}"
)


# ============================================================
# 14. ČUVANJE REZULTATA
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CONTENT-BASED FILTERING - REZULTATI\n"
    )

    f.write(
        "=" * 60 +
        "\n\n"
    )

    f.write(
        f"Ukupno interakcija: "
        f"{len(interactions):,}\n"
    )

    f.write(
        f"Ukupno proizvoda u CBF modelu: "
        f"{len(products):,}\n"
    )

    f.write(
        f"Train recenzija korišćenih za CBF: "
        f"{train_reviews_used:,}\n"
    )

    f.write(
        f"Evaluiranih korisnika: "
        f"{evaluated_users:,}\n"
    )

    f.write(
        f"TOP-K: "
        f"{TOP_K}\n\n"
    )

    f.write(
        f"Test proizvoda poznatih CBF modelu: "
        f"{known_test_percentage:.2f}%\n\n"
    )

    f.write(
        f"Precision@{TOP_K}: "
        f"{precision_at_k:.4f}\n"
    )

    f.write(
        f"Recall@{TOP_K}: "
        f"{recall_at_k:.4f}\n"
    )

    f.write(
        f"Hit Rate@{TOP_K}: "
        f"{hit_rate_at_k:.4f}\n"
    )


print()

print(
    f"Rezultati su sačuvani u: "
    f"{OUTPUT_FILE}"
)

print()

print(
    "CBF evaluacija je završena."
)
