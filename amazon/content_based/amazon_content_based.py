import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


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

OUTPUT_RESULTS = (
    OUTPUT_DIR
    / "amazon_cbf_results.txt"
)

TOP_K = 10

# Bezbedna veličina za laptop sa 8 GB RAM-a
MAX_PRODUCTS = 5_000

# Minimalan broj tekstualnih recenzija
MIN_REVIEWS = 2

# Ograničena TF-IDF dimenzija
MAX_FEATURES = 10_000

MIN_DF = 2
MAX_DF = 0.95

# Broj proizvoda za primere preporuka
SAMPLE_SIZE = 20


# ============================================================
# 1. PRVI PROLAZ - BROJANJE RECENZIJA
# ============================================================

print("Čitanje Amazon skupa...")
print()
print("PRVI PROLAZ")
print("Brojanje recenzija po proizvodu...")

product_counts = defaultdict(int)

total_reviews = 0

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

        product_id = review.get("asin")

        if not product_id:
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

        product_counts[
            product_id
        ] += 1

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
    f"Ukupno proizvoda sa tekstom: "
    f"{len(product_counts):,}"
)


# ============================================================
# 2. IZBOR PROIZVODA
# ============================================================

valid_products = sorted(
    product_id
    for product_id, count in product_counts.items()
    if count >= MIN_REVIEWS
)

total_valid_products = len(
    valid_products
)

print()

print(
    f"Proizvoda sa najmanje "
    f"{MIN_REVIEWS} recenzije: "
    f"{total_valid_products:,}"
)


valid_products = valid_products[
    :MAX_PRODUCTS
]

selected_products = set(
    valid_products
)

print(
    f"Proizvoda izabranih za CBF: "
    f"{len(valid_products):,}"
)


# ============================================================
# 3. DRUGI PROLAZ - TEKST IZABRANIH PROIZVODA
# ============================================================

print()
print(
    "DRUGI PROLAZ"
)

print(
    "Čuvanje teksta samo za izabrane proizvode..."
)

product_texts = defaultdict(list)

used_reviews = 0
processed_reviews = 0

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

        processed_reviews += 1

        product_id = review.get("asin")

        if (
            not product_id
            or product_id not in selected_products
        ):
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
            product_id
        ].append(text)

        used_reviews += 1

        if processed_reviews % 500_000 == 0:

            print(
                f"Obrađeno recenzija: "
                f"{processed_reviews:,}"
            )


print()

print(
    f"Recenzija korišćenih za CBF: "
    f"{used_reviews:,}"
)

print(
    f"Proizvoda sa sačuvanim tekstom: "
    f"{len(product_texts):,}"
)


# ============================================================
# 4. FORMIRANJE TEKSTA PO PROIZVODU
# ============================================================

products = []
texts = []

for product_id in valid_products:

    reviews = product_texts.get(
        product_id,
        []
    )

    if not reviews:
        continue

    combined_text = " ".join(
        reviews
    )

    products.append(
        product_id
    )

    texts.append(
        combined_text
    )


print()

print(
    f"Proizvoda koji ulaze u CBF model: "
    f"{len(products):,}"
)


if not products:

    raise RuntimeError(
        "Nema proizvoda sa tekstualnim sadržajem."
    )


# ============================================================
# 5. OSLOBAĐANJE NEPOTREBNIH PODATAKA
# ============================================================

del product_texts
del product_counts
del selected_products


# ============================================================
# 6. TF-IDF
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
    f"{tfidf_matrix.shape[0]:,} x "
    f"{tfidf_matrix.shape[1]:,}"
)


del texts


# ============================================================
# 7. MODEL ZA SLIČNE PROIZVODE
# ============================================================

print()
print(
    "Priprema modela za pronalaženje "
    "sličnih proizvoda..."
)

nn_model = NearestNeighbors(
    n_neighbors=TOP_K + 1,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

nn_model.fit(
    tfidf_matrix
)

print(
    "Model za pretragu sličnih proizvoda je spreman."
)


# ============================================================
# 8. TESTIRANJE PREPORUKA
# ============================================================

print()
print(
    "Testiranje Content-Based preporuka..."
)

sample_indices = np.linspace(
    0,
    len(products) - 1,
    num=min(
        SAMPLE_SIZE,
        len(products)
    ),
    dtype=int
)


recommendation_examples = []

similarity_values = []


for counter, index in enumerate(
    sample_indices,
    start=1
):

    distances, indices = nn_model.kneighbors(
        tfidf_matrix[index]
    )

    recommendations = []

    for distance, neighbor_index in zip(
        distances[0],
        indices[0]
    ):

        # Preskačemo sam proizvod.
        if neighbor_index == index:
            continue

        similarity = 1.0 - float(
            distance
        )

        recommendations.append(
            (
                products[neighbor_index],
                similarity
            )
        )

        similarity_values.append(
            similarity
        )

        if len(recommendations) >= TOP_K:
            break

    recommendation_examples.append(
        (
            products[index],
            recommendations
        )
    )

    print(
        f"Testirano proizvoda: "
        f"{counter:,}/"
        f"{len(sample_indices):,}"
    )


# ============================================================
# 9. STATISTIKA SLIČNOSTI
# ============================================================

if similarity_values:

    mean_similarity = float(
        np.mean(
            similarity_values
        )
    )

    min_similarity = float(
        np.min(
            similarity_values
        )
    )

    max_similarity = float(
        np.max(
            similarity_values
        )
    )

else:

    mean_similarity = 0.0
    min_similarity = 0.0
    max_similarity = 0.0


# ============================================================
# 10. ČUVANJE REZULTATA
# ============================================================

with open(
    OUTPUT_RESULTS,
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
        f"Ukupno obrađenih recenzija: "
        f"{total_reviews:,}\n"
    )

    f.write(
        f"Proizvoda sa najmanje "
        f"{MIN_REVIEWS} recenzije: "
        f"{total_valid_products:,}\n"
    )

    f.write(
        f"Recenzija korišćenih za CBF: "
        f"{used_reviews:,}\n"
    )

    f.write(
        f"Proizvoda u CBF modelu: "
        f"{len(products):,}\n"
    )

    f.write(
        f"TF-IDF karakteristika: "
        f"{tfidf_matrix.shape[1]:,}\n"
    )

    f.write(
        f"Broj preporuka po proizvodu: "
        f"{TOP_K}\n"
    )

    f.write(
        f"Broj proizvoda korišćenih za primere: "
        f"{len(sample_indices):,}\n\n"
    )

    f.write(
        "STATISTIKA SLIČNOSTI\n"
    )

    f.write(
        "-" * 60 +
        "\n"
    )

    f.write(
        f"Prosečna cosine similarity: "
        f"{mean_similarity:.4f}\n"
    )

    f.write(
        f"Minimalna cosine similarity: "
        f"{min_similarity:.4f}\n"
    )

    f.write(
        f"Maksimalna cosine similarity: "
        f"{max_similarity:.4f}\n\n"
    )

    f.write(
        "PRIMERI PREPORUKA\n"
    )

    f.write(
        "-" * 60 +
        "\n\n"
    )

    for (
        product_id,
        recommendations
    ) in recommendation_examples:

        f.write(
            f"Proizvod: {product_id}\n"
        )

        for (
            recommended_product,
            similarity
        ) in recommendations:

            f.write(
                f"  -> {recommended_product} "
                f"(sličnost: {similarity:.4f})\n"
            )

        f.write("\n")


# ============================================================
# 11. ISPIS
# ============================================================

print()

print(
    "=" * 60
)

print(
    "CONTENT-BASED FILTERING"
)

print(
    "=" * 60
)

print(
    f"Proizvoda u modelu: "
    f"{len(products):,}"
)

print(
    f"TF-IDF karakteristika: "
    f"{tfidf_matrix.shape[1]:,}"
)

print(
    f"Prosečna cosine similarity: "
    f"{mean_similarity:.4f}"
)

print(
    f"Minimalna cosine similarity: "
    f"{min_similarity:.4f}"
)

print(
    f"Maksimalna cosine similarity: "
    f"{max_similarity:.4f}"
)

print()

print(
    f"Rezultati su sačuvani u: "
    f"{OUTPUT_RESULTS}"
)

print()

print(
    "CBF osnovni eksperiment je završen."
)
