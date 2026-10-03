import pandas as pd
import math
from collections import defaultdict, Counter
from itertools import combinations
from heapq import nlargest
from pathlib import Path


# ============================================================
# PODEŠAVANJA
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent

INPUT_FILE = OUTPUT_DIR / "amazon_cf_interactions.csv"
OUTPUT_FILE = OUTPUT_DIR / "amazon_cf_results.txt"

TOP_K = 10
MAX_NEIGHBORS = 50

# Za evaluaciju koristimo poslednje 2 interakcije
# korisnika kao test skup.
N_TEST_ITEMS = 2

# Korisnik mora imati najmanje 5 pozitivnih interakcija
# da bi učestvovao u evaluaciji.
MIN_TOTAL_INTERACTIONS = 5


# ============================================================
# 1. UČITAVANJE PODATAKA
# ============================================================

print("Učitavanje Amazon CF podataka...")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    f"Ukupno interakcija: "
    f"{len(df):,}"
)

print(
    f"Korisnika: "
    f"{df['user_id'].nunique():,}"
)

print(
    f"Proizvoda: "
    f"{df['item_id'].nunique():,}"
)


# ============================================================
# 2. GRUPISANJE INTERAKCIJA PO KORISNIKU
# ============================================================

user_items = defaultdict(list)

for row in df.itertuples(index=False):

    user_items[row.user_id].append(
        (
            row.timestamp,
            row.item_id
        )
    )


# Sortiranje po vremenu
for user in user_items:

    user_items[user].sort(
        key=lambda x: x[0]
    )


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

train_users = {}
test_users = {}

eligible_users = 0

for user, interactions in user_items.items():

    # Uklanjamo eventualne duplikate proizvoda
    # uz očuvanje vremenskog redosleda.
    seen = set()

    unique_interactions = []

    for timestamp, item in interactions:

        if item not in seen:

            seen.add(item)

            unique_interactions.append(
                (
                    timestamp,
                    item
                )
            )

    if len(unique_interactions) < MIN_TOTAL_INTERACTIONS:
        continue

    eligible_users += 1

    # Poslednje 2 interakcije koriste se
    # kao test skup.
    test_part = unique_interactions[
        -N_TEST_ITEMS:
    ]

    train_part = unique_interactions[
        :-N_TEST_ITEMS
    ]

    train_users[user] = [
        item
        for _, item in train_part
    ]

    test_users[user] = [
        item
        for _, item in test_part
    ]


print()
print(
    "TRAIN / TEST PODELA"
)

print(
    "-------------------"
)

print(
    f"Korisnika pogodnih za evaluaciju: "
    f"{eligible_users:,}"
)

print(
    f"Train korisnika: "
    f"{len(train_users):,}"
)

print(
    f"Test korisnika: "
    f"{len(test_users):,}"
)


# ============================================================
# 4. POPULARNOST PROIZVODA U TRAIN SKUPU
# ============================================================

item_user_count = Counter()

for user, items in train_users.items():

    for item in set(items):

        item_user_count[item] += 1


# ============================================================
# 5. FORMIRANJE CO-OCCURRENCE PODATAKA
# ============================================================
#
# Ako korisnik ima:
#
# A, B, C
#
# formiraju se parovi:
#
# A-B
# A-C
# B-C
#
# Ovo koristimo da procenimo koliko se proizvodi
# često pojavljuju zajedno kod istih korisnika.
#

print()
print(
    "Formiranje co-occurrence podataka..."
)

co_occurrence = Counter()

processed_users = 0

for user, items in train_users.items():

    # Uklanjamo duplikate.
    # Sortiranje obezbeđuje determinističan redosled.
    unique_items = sorted(
        set(items)
    )

    # Ne dozvoljavamo ekstremno velike liste
    # koje bi nepotrebno povećale broj kombinacija.
    #
    # Važno:
    # izbor prvih 100 proizvoda je sada deterministički
    # zbog prethodnog sortiranja.
    if len(unique_items) > 100:

        unique_items = unique_items[:100]

    for item_a, item_b in combinations(
        unique_items,
        2
    ):

        co_occurrence[
            (item_a, item_b)
        ] += 1

    processed_users += 1

    if processed_users % 10000 == 0:

        print(
            f"Obrađeno korisnika: "
            f"{processed_users:,}/"
            f"{len(train_users):,}"
        )


print(
    f"Ukupno parova proizvoda: "
    f"{len(co_occurrence):,}"
)


# ============================================================
# 6. COSINE SLIČNOST IZ CO-OCCURRENCE PODATAKA
# ============================================================

print()
print(
    "Izračunavanje sličnosti proizvoda..."
)

neighbors = defaultdict(list)

for (
    item_a,
    item_b
), common_users in co_occurrence.items():

    count_a = item_user_count[item_a]

    count_b = item_user_count[item_b]

    denominator = math.sqrt(
        count_a * count_b
    )

    if denominator == 0:

        continue

    similarity = (
        common_users /
        denominator
    )

    neighbors[item_a].append(
        (
            item_b,
            similarity
        )
    )

    neighbors[item_b].append(
        (
            item_a,
            similarity
        )
    )


# ============================================================
# 7. ZADRŽAVAMO SAMO TOP 50 SUSEDA PO PROIZVODU
# ============================================================

print()
print(
    "Ograničavanje broja suseda po proizvodu..."
)

for item in list(neighbors.keys()):

    neighbors[item] = nlargest(
        MAX_NEIGHBORS,
        neighbors[item],
        key=lambda x: x[1]
    )


print(
    f"Proizvoda sa formiranim susedima: "
    f"{len(neighbors):,}"
)


# ============================================================
# 8. FUNKCIJA ZA PREPORUKE
# ============================================================

def recommend(
    user_items_list,
    top_k=10
):

    scores = defaultdict(float)

    seen_items = set(
        user_items_list
    )

    for item in seen_items:

        for candidate, similarity in neighbors.get(
            item,
            []
        ):

            # Ne preporučujemo proizvod koji je
            # korisnik već imao u train skupu.
            if candidate in seen_items:

                continue

            scores[candidate] += similarity

    if not scores:

        return []

    recommendations = nlargest(
        top_k,
        scores.items(),
        key=lambda x: x[1]
    )

    return recommendations


# ============================================================
# 9. EVALUACIJA
# ============================================================

print()
print(
    "Evaluacija Collaborative Filtering modela..."
)

print(
    "--------------------------------------------"
)

precision_values = []
recall_values = []
hit_values = []

evaluated_users = 0

for user, train_items in train_users.items():

    actual_items = set(
        test_users[user]
    )

    recommendations = recommend(
        train_items,
        TOP_K
    )

    recommended_items = {
        item
        for item, score in recommendations
    }

    hits = len(
        recommended_items.intersection(
            actual_items
        )
    )

    precision = (
        hits /
        TOP_K
    )

    recall = (
        hits /
        len(actual_items)
        if len(actual_items) > 0
        else 0
    )

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


# ============================================================
# 10. PROSEČNE METRIKE
# ============================================================

precision_at_k = (
    sum(precision_values) /
    len(precision_values)
    if precision_values
    else 0
)

recall_at_k = (
    sum(recall_values) /
    len(recall_values)
    if recall_values
    else 0
)

hit_rate_at_k = (
    sum(hit_values) /
    len(hit_values)
    if hit_values
    else 0
)


# ============================================================
# 11. REZULTATI
# ============================================================

print()
print(
    "=" * 55
)

print(
    "COLLABORATIVE FILTERING - REZULTATI"
)

print(
    "=" * 55
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
    f"Broj suseda po proizvodu: "
    f"{MAX_NEIGHBORS}"
)

print()

print(
    f"Precision@{TOP_K}: "
    f"{precision_at_k:.4f}"
)

print(
    f"Recall@{TOP_K}:    "
    f"{recall_at_k:.4f}"
)

print(
    f"Hit Rate@{TOP_K}:  "
    f"{hit_rate_at_k:.4f}"
)


# ============================================================
# 12. ČUVANJE REZULTATA
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "COLLABORATIVE FILTERING - REZULTATI\n"
    )

    f.write(
        "=" * 55 +
        "\n\n"
    )

    f.write(
        f"Ukupno interakcija: "
        f"{len(df):,}\n"
    )

    f.write(
        f"Korisnika: "
        f"{df['user_id'].nunique():,}\n"
    )

    f.write(
        f"Proizvoda: "
        f"{df['item_id'].nunique():,}\n\n"
    )

    f.write(
        f"Evaluiranih korisnika: "
        f"{evaluated_users:,}\n"
    )

    f.write(
        f"TOP-K: "
        f"{TOP_K}\n"
    )

    f.write(
        f"Broj suseda po proizvodu: "
        f"{MAX_NEIGHBORS}\n\n"
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
    "Collaborative Filtering je završen."
)
