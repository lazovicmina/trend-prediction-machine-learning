import gzip
import json
import hashlib
import csv
from collections import defaultdict
from pathlib import Path


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

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "amazon_cf_interactions.csv"
)

USER_SELECTION_MODULO = 7


# ============================================================
# ODABIR KORISNIKA
# ============================================================

def select_user(reviewer_id):
    """
    Deterministički bira korisnike na osnovu hash vrednosti.

    Na ovaj način isti dataset i isti parametri
    uvek daju isti skup korisnika.
    """

    hash_value = hashlib.md5(
        reviewer_id.encode("utf-8")
    ).hexdigest()

    number = int(hash_value, 16)

    return number % USER_SELECTION_MODULO == 0


# ============================================================
# PRIKUPLJANJE INTERAKCIJA
# ============================================================

def collect_interactions(file_path):

    interactions = defaultdict(list)

    total_reviews = 0
    selected_reviews = 0
    selected_users = set()

    print("=" * 60)
    print("PRIPREMA PODATAKA ZA COLLABORATIVE FILTERING")
    print("=" * 60)
    print()

    print(
        f"Originalni dataset: {file_path}"
    )

    print(
        f"Odabir korisnika: hash % {USER_SELECTION_MODULO} == 0"
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

            total_reviews += 1

            reviewer_id = review.get(
                "reviewerID"
            )

            asin = review.get(
                "asin"
            )

            rating = review.get(
                "overall"
            )

            timestamp = review.get(
                "unixReviewTime"
            )

            if not reviewer_id:
                continue

            if not asin:
                continue

            if rating is None:
                continue

            if timestamp is None:
                continue

            # Odabir korisnika
            if not select_user(
                reviewer_id
            ):
                continue

            selected_users.add(
                reviewer_id
            )

            selected_reviews += 1

            interactions[
                reviewer_id
            ].append(
                (
                    asin,
                    float(rating),
                    int(timestamp)
                )
            )

            # Progress
            if total_reviews % 500_000 == 0:

                print(
                    f"Obrađeno: "
                    f"{total_reviews:,} | "
                    f"Odabrane interakcije: "
                    f"{selected_reviews:,} | "
                    f"Korisnici: "
                    f"{len(selected_users):,}"
                )

    return (
        interactions,
        total_reviews
    )


# ============================================================
# ČIŠĆENJE INTERAKCIJA
# ============================================================

def prepare_interactions(
    interactions
):

    prepared = []

    users_with_multiple_items = 0

    positive_interactions = 0

    for user_id, user_interactions in interactions.items():

        # Sortiranje po vremenu
        user_interactions.sort(
            key=lambda x: x[2]
        )

        # Uklanjamo duplikate proizvoda.
        # Ako korisnik ima više recenzija istog proizvoda,
        # zadržavamo poslednju.
        latest_by_product = {}

        for asin, rating, timestamp in user_interactions:

            latest_by_product[asin] = (
                rating,
                timestamp
            )

        unique_interactions = [
            (
                asin,
                rating,
                timestamp
            )
            for asin, (
                rating,
                timestamp
            ) in latest_by_product.items()
        ]

        unique_interactions.sort(
            key=lambda x: x[2]
        )

        if len(unique_interactions) < 2:
            continue

        users_with_multiple_items += 1

        for asin, rating, timestamp in unique_interactions:

            # Za recommendation task koristimo
            # pozitivne interakcije.
            if rating >= 4:

                positive_interactions += 1

                prepared.append(
                    (
                        user_id,
                        asin,
                        rating,
                        timestamp
                    )
                )

    return (
        prepared,
        users_with_multiple_items,
        positive_interactions
    )


# ============================================================
# ČUVANJE
# ============================================================

def save_interactions(
    interactions
):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "user_id",
                "item_id",
                "rating",
                "timestamp"
            ]
        )

        writer.writerows(
            interactions
        )

    print()
    print(
        f"Podaci sačuvani u: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    interactions, total_reviews = (
        collect_interactions(
            FILE_PATH
        )
    )

    (
        prepared,
        users_with_multiple_items,
        positive_interactions
    ) = prepare_interactions(
        interactions
    )

    print()
    print("=" * 60)
    print("REZULTATI PRIPREME")
    print("=" * 60)

    print()
    print(
        f"Ukupno obrađenih recenzija: "
        f"{total_reviews:,}"
    )

    print(
        f"Odabranih korisnika: "
        f"{len(interactions):,}"
    )

    print(
        f"Korisnika sa najmanje 2 proizvoda: "
        f"{users_with_multiple_items:,}"
    )

    print(
        f"Pozitivnih interakcija: "
        f"{positive_interactions:,}"
    )

    print(
        f"Ukupno sačuvanih interakcija: "
        f"{len(prepared):,}"
    )

    save_interactions(
        prepared
    )

    print()
    print("=" * 60)
    print("PRIPREMA ZAVRŠENA.")
    print("=" * 60)
