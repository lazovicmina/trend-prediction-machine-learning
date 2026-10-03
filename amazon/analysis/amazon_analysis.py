import gzip
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


# Putanja do glavnog foldera projekta
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Lokacija Amazon skupa podataka
FILE_PATH = (
    PROJECT_ROOT
    / "data-service"
    / "data"
    / "amazon"
    / "Electronics_5.json.gz"
)


def analyze_dataset(file_path):
    total_reviews = 0

    ratings = Counter()
    users = set()
    products = set()

    review_text_count = 0
    summary_count = 0
    helpful_count = 0

    min_date = None
    max_date = None

    print("Pokretanje analize...")
    print("Fajl:", file_path)
    print()

    with gzip.open(file_path, "rt", encoding="utf-8") as file:

        for line in file:

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_reviews += 1

            # Rating
            rating = review.get("overall")

            if rating is not None:
                ratings[rating] += 1

            # User
            reviewer_id = review.get("reviewerID")

            if reviewer_id:
                users.add(reviewer_id)

            # Product
            asin = review.get("asin")

            if asin:
                products.add(asin)

            # Review text
            if review.get("reviewText"):
                review_text_count += 1

            # Summary
            if review.get("summary"):
                summary_count += 1

            # Helpful votes
            if review.get("helpful"):
                helpful_count += 1

            # Date
            unix_time = review.get("unixReviewTime")

            if unix_time:
                try:
                    date = datetime.fromtimestamp(unix_time).date()

                    if min_date is None or date < min_date:
                        min_date = date

                    if max_date is None or date > max_date:
                        max_date = date

                except (ValueError, OSError, OverflowError):
                    pass

            # Progress
            if total_reviews % 500_000 == 0:
                print(f"Obrađeno: {total_reviews:,} recenzija")

    print("\n" + "=" * 50)
    print("REZULTATI ANALIZE")
    print("=" * 50)

    print(f"\nUkupan broj recenzija: {total_reviews:,}")
    print(f"Broj korisnika: {len(users):,}")
    print(f"Broj proizvoda: {len(products):,}")

    print(f"\nRecenzije sa tekstom: {review_text_count:,}")
    print(f"Recenzije sa sažetkom: {summary_count:,}")
    print(f"Recenzije sa helpful podacima: {helpful_count:,}")

    print("\nDistribucija ocena:")

    for rating in sorted(ratings):
        count = ratings[rating]
        percentage = count / total_reviews * 100

        print(
            f"  {rating} zvezdica: "
            f"{count:,} ({percentage:.2f}%)"
        )

    print("\nVremenski period:")

    if min_date:
        print(f"  Od: {min_date}")

    if max_date:
        print(f"  Do: {max_date}")

    print("\nAnaliza završena.")


if __name__ == "__main__":
    analyze_dataset(FILE_PATH)
