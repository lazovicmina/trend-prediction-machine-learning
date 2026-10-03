import os
import requests
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


DATA_SERVIS_URL = os.getenv(
    "DATA_SERVIS_URL",
    "http://127.0.0.1:8002/dataset"
)

PUTANJA_MODELA = "app/classification_models.joblib"
PUTANJA_METRIKA = "app/classification_metrics.joblib"


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
    """
    Učitavanje pripremljenog Walmart skupa
    podataka iz Data Service-a.
    """

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

    # Hronološko sortiranje.
    df = df.sort_values(
        by="Date"
    ).reset_index(drop=True)

    x = df[FEATURES].copy()

    if "IsHoliday" in x.columns:
        x["IsHoliday"] = x["IsHoliday"].astype(int)

    return df, x


def napravi_klasifikacioni_model(ime_modela: str):
    """
    Kreiranje izabranog klasifikacionog modela.
    """

    if ime_modela == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )

    if ime_modela == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2
        )

    raise ValueError(
        "Nepoznat model. Koristite "
        "'decision_tree' ili 'xgboost'."
    )


def napravi_ciljnu_promenljivu(
    df: pd.DataFrame,
    train_mask: pd.Series
):
    """
    Formiranje ciljne promenljive.

    Posmatranje se označava kao popularno (1)
    ako je Weekly_Sales veći ili jednak
    medijani prodaje iz trening perioda.

    Prag se računa isključivo na trening podacima
    kako bi se izbeglo curenje informacija iz test skupa.
    """

    prag_popularnosti = float(
        df.loc[
            train_mask,
            "Weekly_Sales"
        ].median()
    )

    popular = (
        df["Weekly_Sales"] >= prag_popularnosti
    ).astype(int)

    return popular, prag_popularnosti


def treniraj_klasifikacioni_model(
    ime_modela: str = "decision_tree"
):
    """
    Treniranje klasifikacionog modela i evaluacija
    na hronološki podeljenom Walmart skupu podataka.
    """

    df, x = ucitaj_podatke()

    broj_redova = len(df)

    if broj_redova < 10:
        raise ValueError(
            "Nema dovoljno podataka za treniranje modela."
        )

    # 80% najstarijih podataka za treniranje,
    # 20% najnovijih podataka za testiranje.
    granica = int(broj_redova * 0.8)

    datum_podele = df.iloc[granica]["Date"]

    train_mask = df["Date"] < datum_podele
    test_mask = df["Date"] >= datum_podele

    x_train = x.loc[train_mask].copy()
    x_test = x.loc[test_mask].copy()

    if x_train.empty or x_test.empty:
        raise ValueError(
            "Nije moguće napraviti validnu hronološku podelu."
        )

    # Formiranje ciljne promenljive.
    # Prag se računa samo iz trening perioda.
    y, prag_popularnosti = napravi_ciljnu_promenljivu(
        df,
        train_mask
    )

    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    # Provera da obe klase postoje u trening skupu.
    if y_train.nunique() < 2:
        raise ValueError(
            "Trening skup nema obe klase popularnosti."
        )

    model = napravi_klasifikacioni_model(
        ime_modela
    )

    model.fit(
        x_train,
        y_train
    )

    predictions = model.predict(
        x_test
    )

    accuracy = float(
        accuracy_score(
            y_test,
            predictions
        )
    )

    precision = float(
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    recall = float(
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    f1 = float(
        f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    train_positive = int(
        y_train.sum()
    )

    train_negative = int(
        len(y_train) - train_positive
    )

    test_positive = int(
        y_test.sum()
    )

    test_negative = int(
        len(y_test) - test_positive
    )

    rezultat = {
        "model_name": ime_modela,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix.tolist(),
        "popularity_threshold": prag_popularnosti,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "train_positive": train_positive,
        "train_negative": train_negative,
        "test_positive": test_positive,
        "test_negative": test_negative,
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

    # Učitavanje postojećih modela ako postoje,
    # kako se Decision Tree i XGBoost čuvaju zajedno.
    modeli = {}

    if os.path.exists(PUTANJA_MODELA):
        modeli = joblib.load(
            PUTANJA_MODELA
        )

    modeli[ime_modela] = {
        "model": model,
        "features": FEATURES,
        "popularity_threshold": prag_popularnosti
    }

    joblib.dump(
        modeli,
        PUTANJA_MODELA
    )

    # Čuvanje rezultata za izabrani model.
    metrike = {}

    if os.path.exists(PUTANJA_METRIKA):
        metrike = joblib.load(
            PUTANJA_METRIKA
        )

    metrike[ime_modela] = rezultat

    joblib.dump(
        metrike,
        PUTANJA_METRIKA
    )

    return {
        "poruka": "Klasifikacioni model je istreniran",
        **rezultat
    }
