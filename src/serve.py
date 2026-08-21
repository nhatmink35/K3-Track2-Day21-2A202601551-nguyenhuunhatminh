from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

AWS_BUCKET = os.environ.get("CLOUD_BUCKET", os.environ.get("AWS_BUCKET", ""))
AWS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu AWS S3 ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import.
    Su dung AWS credentials mac dinh (~/.aws/credentials, bien moi truong hoac IAM role tren EC2).
    """
    # TODO 1: Khoi tao boto3 client cho S3
    s3 = boto3.client("s3")

    # TODO 2: Tao thu muc luu file neu chua co
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # TODO 3: Tai file model xuong may tu S3 bucket
    s3.download_file(AWS_BUCKET, AWS_MODEL_KEY, MODEL_PATH)

    # TODO 4: In thong bao thanh cong
    print(f"Model da duoc tai xuong tu S3: s3://{AWS_BUCKET}/{AWS_MODEL_KEY} -> {MODEL_PATH}")


# Goi ham nay khi module duoc import (chay khi server khoi dong)
if AWS_BUCKET:
    try:
        download_model()
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Khong the tai model tu S3: {e}")
        model = None
elif os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    # TODO 5: Tra ve dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    # TODO 6: Kiem tra so luong dac trung (phai bang 12).
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=500, detail="Model is not loaded")

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    pred = int(model.predict([req.features])[0])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": pred, "label": label_map.get(pred, "unknown")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
