from fastapi import APIRouter, HTTPException
import os
import requests


router = APIRouter()


MODEL_SERVIS_URL = os.getenv(
    "MODEL_SERVIS_URL",
    "http://127.0.0.1:8001"
)

REQUEST_TIMEOUT = 300


def pozovi_model_servis(
    method: str,
    path: str,
    **kwargs
):
    url = f"{MODEL_SERVIS_URL}{path}"

    try:
        response = requests.request(
            method,
            url,
            timeout=REQUEST_TIMEOUT,
            **kwargs
        )

        if not response.ok:
            try:
                detalji = response.json()
            except ValueError:
                detalji = response.text

            raise HTTPException(
                status_code=response.status_code,
                detail=detalji
            )

        return response.json()

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Trend Service nije odgovorio na vreme."
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Trend Service trenutno nije dostupan."
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Greška u komunikaciji sa "
                f"Trend Service-om: {str(e)}"
            )
        )


# ---------------------------------------------------------
# REGRESIONI MODELI
# ---------------------------------------------------------

@router.post("/modeli/treniraj")
def treniraj_model(
    model: str = "linear"
):
    if model not in [
        "linear",
        "rf"
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model mora biti "
                "'linear' ili 'rf'."
            )
        )

    return pozovi_model_servis(
        "POST",
        "/modeli/treniraj",
        params={
            "model": model
        }
    )


@router.get("/modeli/metrike")
def vrati_metrike():
    return pozovi_model_servis(
        "GET",
        "/modeli/metrike"
    )


@router.post("/modeli/predvidi")
def predvidi(payload: dict):
    return pozovi_model_servis(
        "POST",
        "/modeli/predvidi",
        json=payload
    )


# ---------------------------------------------------------
# KLASIFIKACIONI MODELI
# ---------------------------------------------------------

@router.post("/klasifikacija/treniraj")
def treniraj_klasifikaciju(
    model: str = "decision_tree"
):
    if model not in [
        "decision_tree",
        "xgboost"
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model mora biti "
                "'decision_tree' ili 'xgboost'."
            )
        )

    return pozovi_model_servis(
        "POST",
        "/klasifikacija/treniraj",
        params={
            "model": model
        }
    )


@router.get("/klasifikacija/metrike")
def vrati_klasifikacione_metrike():
    return pozovi_model_servis(
        "GET",
        "/klasifikacija/metrike"
    )
