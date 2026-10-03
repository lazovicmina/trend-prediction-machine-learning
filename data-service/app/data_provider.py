import pandas as pd
from pathlib import Path


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "walmart"
    / "train.csv"
)


def get_dataset():
    df = pd.read_csv(DATA_PATH)

    # Pretvaranje datuma
    df["Date"] = pd.to_datetime(df["Date"])

    # Sortiranje je važno za formiranje istorijskih karakteristika.
    df = df.sort_values(
        by=["Store", "Dept", "Date"]
    ).copy()

    # Vremenske karakteristike
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = (
        df["Date"].dt.isocalendar().week.astype(int)
    )
    df["day_of_week"] = df["Date"].dt.dayofweek

    # Grupisanje po prodavnici i odeljenju
    group = df.groupby(
        ["Store", "Dept"]
    )["Weekly_Sales"]

    # Prodaja iz prethodne nedelje
    df["sales_lag_1"] = group.shift(1)

    # Prodaja od pre dve nedelje
    df["sales_lag_2"] = group.shift(2)

    # Prosek prodaje prethodnih 7 nedelja.
    # Računa se posebno za svaki Store + Dept.
    # shift(1) obezbeđuje da trenutna prodaja
    # nije uključena u prosek.
    df["sales_previous_7_weeks"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    # Uklanjanje redova za koje nema potrebnih
    # istorijskih karakteristika.
    df = df.dropna(
        subset=[
            "sales_lag_1",
            "sales_lag_2",
            "sales_previous_7_weeks"
        ]
    )

    return df
