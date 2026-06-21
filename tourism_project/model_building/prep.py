import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# CHANGE THIS: your HF username and dataset repo
DATASET_PATH = "hf://datasets/prakashpabba/tourism-project/tourism.csv"
#DATASET_PATH = "hf://datasets/prakashpabba/tourism-dataset/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Drop unique identifier
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# Convert nominal categorical columns to dummy variables
cat_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

target_col = "ProdTaken"

X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        #repo_id="prakashpabba/tourism-dataset",
        repo_id="prakashpabba/tourism-project",
        repo_type="dataset",
    )

print("Train/test splits uploaded to Hugging Face dataset repo.")
