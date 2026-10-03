import pandas as pd
from collections import defaultdict
from itertools import combinations
from heapq import nlargest
import math
from pathlib import Path


# ============================================================
# PODEŠAVANJA
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    OUTPUT_DIR
    / "amazon_cf_interactions.csv"
)

TOP_K = 10
MAX_NEIGHBORS = 50
N_TEST_ITEMS = 2
MIN_TOTAL_INTERACTIONS = 5


# ============================================================
# UČITAVANJE PODATAKA
# ============================================================

print("Učitavanje podataka...")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    f"Interakcija: "
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
# 1. GRUPISANJE PO KORISNIKU
# ============================================================

user_items = defaultdict(list)

for row in df.itertuples(index=False):

    user_items[row.user_id].append(
        (
            row.timestamp,
            row.item_id
        )
    )


for user in user_items:

    user_items[user].sort(
        key=lambda x: x[0]
    )


# ============================================================
# 2. TRAIN / TEST
# ============================================================

train_users = {}
test_users = {}

for user, interactions in user_items.items():

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

    train_part = unique_interactions[
        :-N_TEST_ITEMS
    ]

    test_part = unique_interactions[
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


print()

print(
    "Korisnici za evaluaciju:",
    f"{len(train_users):,}"
)


# ============================================================
# 3. PROIZVODI DOSTUPNI U TRAIN SKUPU
# ============================================================

train_items = set()

for items in train_users.values():

    train_items.update(items)


print()

print("TRAIN ANALIZA")
print("----------------------------")

print(
    "Proizvoda prisutnih u train skupu:",
    f"{len(train_items):,}"
)


# ============================================================
# 4. CO-OCCURRENCE
# ============================================================

print()

print(
    "Formiranje co-occurrence podataka..."
)

item_user_count = {}

for items in train_users.values():

    for item in set(items):

        item_user_count[item] = (
            item_user_count.get(
                item,
                0
            ) + 1
        )


co_occurrence = {}

processed = 0

for items in train_users.values():

    unique_items = sorted(
        set(items)
    )

    for item_a, item_b in combinations(
        unique_items,
        2
    ):

        pair = (
            item_a,
            item_b
        )

        co_occurrence[pair] = (
            co_occurrence.get(
                pair,
                0
            ) + 1
        )

    processed += 1

    if processed % 10000 == 0:

        print(
            f"Obrađeno: "
            f"{processed:,}/"
            f"{len(train_users):,}"
        )


print(
    "Broj parova:",
    f"{len(co_occurrence):,}"
)


# ============================================================
# 5. FORMIRANJE SUSEDA
# ============================================================

neighbors = defaultdict(list)

for (
    item_a,
    item_b
), common_users in co_occurrence.items():

    denominator = math.sqrt(
        item_user_count[item_a]
        *
        item_user_count[item_b]
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


for item in list(neighbors.keys()):

    neighbors[item] = nlargest(
        MAX_NEIGHBORS,
        neighbors[item],
        key=lambda x: x[1]
    )


# ============================================================
# 6. ANALIZA TEST PROIZVODA
# ============================================================

test_items_total = 0
test_items_in_train = 0
test_items_with_neighbors = 0

users_with_unknown_test_item = 0
users_with_recommendations = 0

recommendation_counts = []


def recommend(user_items_list):

    scores = defaultdict(float)

    seen = set(
        user_items_list
    )

    for item in seen:

        for candidate, similarity in neighbors.get(
            item,
            []
        ):

            if candidate in seen:
                continue

            scores[candidate] += similarity

    if not scores:
        return []

    return nlargest(
        TOP_K,
        scores.items(),
        key=lambda x: x[1]
    )


# ============================================================
# 7. EVALUACIJA POKRIVENOSTI
# ============================================================

for user, test_items in test_users.items():

    known_test = False
    known_neighbor_test = False

    for item in test_items:

        test_items_total += 1

        if item in train_items:

            test_items_in_train += 1

            known_test = True

        if item in neighbors:

            test_items_with_neighbors += 1

            known_neighbor_test = True

    if not known_test:

        users_with_unknown_test_item += 1

    recommendations = recommend(
        train_users[user]
    )

    if recommendations:

        users_with_recommendations += 1

    recommendation_counts.append(
        len(recommendations)
    )


# ============================================================
# 8. REZULTATI
# ============================================================

print()

print("=" * 60)

print(
    "COLLABORATIVE FILTERING - DIJAGNOSTIKA"
)

print("=" * 60)

print()

print("TEST PROIZVODI")
print("----------------------------")

print(
    f"Ukupno test proizvoda: "
    f"{test_items_total:,}"
)

print(
    f"Test proizvoda prisutnih u train skupu: "
    f"{test_items_in_train:,}"
)

print(
    f"Procenat poznatih test proizvoda: "
    f"{test_items_in_train / test_items_total * 100:.2f}%"
)

print(
    f"Test proizvoda sa formiranim susedima: "
    f"{test_items_with_neighbors:,}"
)

print(
    f"Procenat test proizvoda sa susedima: "
    f"{test_items_with_neighbors / test_items_total * 100:.2f}%"
)


print()

print("KORISNICI")
print("----------------------------")

print(
    f"Ukupno korisnika za evaluaciju: "
    f"{len(test_users):,}"
)

print(
    f"Korisnika bez poznatog test proizvoda: "
    f"{users_with_unknown_test_item:,}"
)

print(
    f"Korisnika kojima je generisana preporuka: "
    f"{users_with_recommendations:,}"
)

print(
    f"Procenat korisnika sa preporukama: "
    f"{users_with_recommendations / len(test_users) * 100:.2f}%"
)

print()

print("=" * 60)

print(
    "DIJAGNOSTIKA ZAVRŠENA"
)

print("=" * 60)
