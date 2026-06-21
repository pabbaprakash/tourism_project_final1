import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow


from huggingface_hub import hf_hub_download
import pandas as pd

repo_id = "prakashpabba/tourism-project"

#Xtrain_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="Xtrain.csv")
#Xtest_file  = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="Xtest.csv")
#ytrain_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="ytrain.csv")
#ytest_file  = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="ytest.csv")

#Xtrain = pd.read_csv(Xtrain_file)
#Xtest = pd.read_csv(Xtest_file)
#ytrain = pd.read_csv(ytrain_file).squeeze()
#ytest = pd.read_csv(ytest_file).squeeze()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# CHANGE THIS: your HF username and dataset repo
base_repo = "prakashpabba/tourism-project"
Xtrain_path = f"hf://datasets/{base_repo}/Xtrain.csv"
Xtest_path = f"hf://datasets/{base_repo}/Xtest.csv"
ytrain_path = f"hf://datasets/{base_repo}/ytrain.csv"
ytest_path = f"hf://datasets/{base_repo}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

numeric_features = [
    "Age",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]
categorical_features = ["CityTier"]

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, eval_metric="logloss"
)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    classification_threshold = 0.45
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics(
        {
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1-score": test_report["1"]["f1-score"],
        }
    )

    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")

    model_repo_id = "prakashpabba/tourism-model"
    model_repo_type = "model"

    try:
        api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
        print(f"Model repo '{model_repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Model repo '{model_repo_id}' not found. Creating...")
        create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
        print(f"Model repo '{model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=model_repo_id,
        repo_type=model_repo_type,
    )
    print("Model uploaded to Hugging Face.")
