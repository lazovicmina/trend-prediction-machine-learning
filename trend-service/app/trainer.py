import os
import requests
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_SERVIS_URL = os.getenv(
    "DATA_SERVIS_URL",
    "http://127.0.0.1:8002/dataset"
)

PUTANJA_MODELA = "app/model.joblib"
PUTANJA_METRIKA = "app/metrics.joblib"


FEATURES = [
    "Store",
    "Dept",
    "IsHoliday",
    "year",
    "month",
    "week_of_year",
    "day_of_week",
    "sales_lag_1",
    "sales_lag_2",
    "sales_previous_7_weeks"
]


def ucitaj_podatke():
    response = requests.get(
        DATA_SERVIS_URL,
        timeout=120
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json())

    if df.empty:
        raise ValueError(
            "Data Service je vratio prazan skup podataka."
        )

    required_columns = [
        "Date",
        "Weekly_Sales",
        *FEATURES
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Nedostaju kolone: "
            + ", ".join(missing_columns)
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        by="Date"
    ).reset_index(drop=True)

    y = df["Weekly_Sales"].copy()
    x = df[FEATURES].copy()

    if "IsHoliday" in x.columns:
        x["IsHoliday"] = x["IsHoliday"].astype(int)

    return df, x, y


def napravi_model(ime_modela: str):

    if ime_modela == "linear":
        return Pipeline([
            (
                "scaler",
                StandardScaler(with_mean=False)
            ),
            (
                "model",
                LinearRegression()
            )
        ])

    if ime_modela == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=1
        )

    raise ValueError(
        "Nepoznat model. Koristite 'linear' ili 'rf'."
    )


def treniraj_model(ime_modela: str = "linear"):

    df, x, y = ucitaj_podatke()

    broj_redova = len(df)

    if broj_redova < 10:
        raise ValueError(
            "Nema dovoljno podataka za treniranje modela."
        )

    granica = int(broj_redova * 0.8)
    datum_podele = df.iloc[granica]["Date"]

    train_mask = df["Date"] < datum_podele
    test_mask = df["Date"] >= datum_podele

    x_train = x.loc[train_mask].copy()
    x_test = x.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    if x_train.empty or x_test.empty:
        raise ValueError(
            "Nije moguće napraviti validnu hronološku podelu."
        )

    model = napravi_model(ime_modela)

    model.fit(
        x_train,
        y_train
    )

    preds = model.predict(
        x_test
    )

    mae = float(
        mean_absolute_error(
            y_test,
            preds
        )
    )

    rmse = float(
        mean_squared_error(
            y_test,
            preds
        ) ** 0.5
    )

    r2 = float(
        r2_score(
            y_test,
            preds
        )
    )

    joblib.dump(
        {
            "model_name": ime_modela,
            "features": FEATURES,
            "model": model
        },
        PUTANJA_MODELA
    )

    joblib.dump(
        {
            "model_name": ime_modela,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "train_start": str(
                df.loc[
                    train_mask,
                    "Date"
                ].min().date()
            ),
            "train_end": str(
                df.loc[
                    train_mask,
                    "Date"
                ].max().date()
            ),
            "test_start": str(
                df.loc[
                    test_mask,
                    "Date"
                ].min().date()
            ),
            "test_end": str(
                df.loc[
                    test_mask,
                    "Date"
                ].max().date()
            )
        },
        PUTANJA_METRIKA
    )

    return {
        "poruka": "Model istreniran",
        "model": ime_modela,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "train_start": str(
            df.loc[
                train_mask,
                "Date"
            ].min().date()
        ),
        "train_end": str(
            df.loc[
                train_mask,
                "Date"
            ].max().date()
        ),
        "test_start": str(
            df.loc[
                test_mask,
                "Date"
            ].min().date()
        ),
        "test_end": str(
            df.loc[
                test_mask,
                "Date"
            ].max().date()
        )
    }
