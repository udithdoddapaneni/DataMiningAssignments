# GiG


from io import StringIO

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from ds5612_pa2.code import pipeline_configs


app = FastAPI()


class PredictRequest(BaseModel):
    """PredictRequest is a simple request body asking for a classifier and features."""

    classifier: pipeline_configs.ValidClassifierNames
    features: list[float]


class PredictionResponse(BaseModel):
    """PredictionResponse is the response type for API V1."""

    predicted_class: int
    ml_model_version: str = "V1"


class DetailedPredictionResponse(PredictionResponse):
    """DetailedPredictionResponse is the response type for API V2."""

    ml_model_version: str = "V2"
    probabilities: tuple[float, float]


@app.post("/v1/predict")
def predict_v1(request: PredictRequest) -> PredictionResponse:
    # pass the right params from PredictRequest
    #  and create PredictionResponse appropriately
    try:
        prediction = pipeline_configs.get_prediction_class(request.features, request.classifier)
        return PredictionResponse(predicted_class=prediction)
    except:
        raise HTTPException(422)


@app.post("/v2/predict")
def predict_v2(request: PredictRequest) -> DetailedPredictionResponse:
    # pass the right params from PredictRequest
    #  and create DetailedPredictionResponse appropriately
    try:
        ml_pipeline = pipeline_configs.get_simple_ml_pipeline(request.classifier)
        ml_pipeline.train()

        # pass the right params from PredictRequest
        probabilities = ml_pipeline.get_prediction_probabilities(request.features)
        prediction = ml_pipeline.get_prediction_class(request.features)
        return DetailedPredictionResponse(predicted_class=prediction, probabilities=probabilities)
    except:
        raise HTTPException(422)


@app.post("/batch_predict/")
def batch_predict(input_file: UploadFile) -> list[DetailedPredictionResponse]:
    # Hard code classifier as decision tree
    try:
        ml_pipeline = pipeline_configs.get_simple_ml_pipeline(
            pipeline_configs.ValidClassifierNames.DT
        )
        ml_pipeline.train()

        # Change the logic from below.
        # Parse the input_file, get predictions using the code in predict_v2 as a sample
        #  return the list of DetailedPredictionResponse
        inputs = input_file.file.read().decode().split("\n")
        input_features = [list(map(float, i.split())) for i in inputs]
        probabilities = [
            ml_pipeline.get_prediction_probabilities(features) for features in input_features
        ]
        prediction = [ml_pipeline.get_prediction_class(features) for features in input_features]
        output = [
            DetailedPredictionResponse(predicted_class=i, probabilities=j)
            for i, j in zip(prediction, probabilities, strict=False)
        ]
        return output
    except:
        raise HTTPException(422)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
