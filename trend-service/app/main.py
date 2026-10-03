from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import joblib
import os

from app.trainer import treniraj_model
from app.classification_trainer import (
    treniraj_klasifikacioni_model
)


app = FastAPI(
    title="Servis za predviđanje trendova",
    version="6.0.0"
)


PUTANJA_MODELA = "app/model.joblib"
PUTANJA_METRIKA = "app/metrics.joblib"


class ZahtevPredikcije(BaseModel):
    store: int = Field(..., ge=1)
    dept: int = Field(..., ge=1)
    is_holiday: int = Field(0, ge=0, le=1)

    year: int = Field(..., ge=1900)
    month: int = Field(..., ge=1, le=12)
    week_of_year: int = Field(..., ge=1, le=53)
    day_of_week: int = Field(..., ge=0, le=6)

    sales_lag_1: float = Field(..., ge=0)
    sales_lag_2: float = Field(..., ge=0)
    sales_previous_7_weeks: float = Field(..., ge=0)


class OdgovorPredikcije(BaseModel):
    predicted_sales: float


@app.get("/")
def status():
    return {
        "servis": "trend-service",
        "status": "radi"
    }


# ---------------------------------------------------------
# REGRESIONI MODELI
# ---------------------------------------------------------

@app.post("/modeli/treniraj")
def treniraj(
    model: str = Query(
        "linear",
        pattern="^(linear|rf)$"
    )
):
    try:
        return treniraj_model(model)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/modeli/metrike")
def metrike():

    if not os.path.exists(PUTANJA_METRIKA):
        raise HTTPException(
            status_code=400,
            detail="Model još nije treniran."
        )

    return joblib.load(
        PUTANJA_METRIKA
    )


@app.post("/modeli/predvidi")
def predvidi(req: ZahtevPredikcije):

    if not os.path.exists(PUTANJA_MODELA):
        raise HTTPException(
            status_code=400,
            detail=(
                "Model nije treniran. "
                "Pokrenite /modeli/treniraj."
            )
        )

    sacuvan = joblib.load(
        PUTANJA_MODELA
    )

    model = sacuvan["model"]

    X = [[
        req.store,
        req.dept,
        req.is_holiday,
        req.year,
        req.month,
        req.week_of_year,
        req.day_of_week,
        req.sales_lag_1,
        req.sales_lag_2,
        req.sales_previous_7_weeks
    ]]

    try:
        pred = model.predict(X)[0]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Greška prilikom predikcije: "
                f"{str(e)}"
            )
        )

    return {
        "predicted_sales": float(pred)
    }


# ---------------------------------------------------------
# KLASIFIKACIONI MODELI
# ---------------------------------------------------------

@app.post("/klasifikacija/treniraj")
def treniraj_klasifikaciju(
    model: str = Query(
        "decision_tree",
        pattern="^(decision_tree|xgboost)$"
    )
):
    try:
        return treniraj_klasifikacioni_model(
            model
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/klasifikacija/metrike")
def klasifikacione_metrike():

    putanja = "app/classification_metrics.joblib"

    if not os.path.exists(putanja):
        raise HTTPException(
            status_code=400,
            detail=(
                "Klasifikacioni model još nije treniran."
            )
        )

    return joblib.load(
        putanja
    )
