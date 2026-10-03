from fastapi import FastAPI, Query
from app.data_provider import get_dataset


app = FastAPI(
    title="Data Service",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "service": "data-service",
        "status": "running"
    }


@app.get("/dataset")
def dataset(
    limit: int | None = Query(
        None,
        ge=1,
        description="Maximum number of records returned. "
                    "If omitted, the complete dataset is returned."
    )
):
    df = get_dataset()

    if limit is not None:
        df = df.head(limit)

    return df.to_dict(orient="records")
