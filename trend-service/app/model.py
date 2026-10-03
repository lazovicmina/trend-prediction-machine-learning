import joblib


def ucitaj_model(putanja_modela: str):
    """
    Učitava prethodno sačuvan model.
    """
    sacuvan = joblib.load(putanja_modela)
    return sacuvan["model"]


def predvidi_prodaju(
    model,
    karakteristike
):
    """
    Vrši predikciju količine prodaje
    na osnovu karakteristika proizvoda.
    """
    predikcija = model.predict(
        [karakteristike]
    )[0]

    return float(predikcija)
