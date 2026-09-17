import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COLUMN = "Churn"


def load_raw_data(filepath=None):
    if filepath is None:
        filepath = DATA_PATH
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    df = df.drop(columns=["customerID"], errors="ignore")

    df["TotalCharges"] = df["TotalCharges"].replace({" ": 0.0})
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return df


def encode_target(df):
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({"Yes": 1, "No": 0})
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def encode_features(df):
    df = df.copy()
    encoders = {}
    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def preprocess_data(df=None, test_size=0.2, random_state=RANDOM_STATE):
    if df is None:
        df = load_raw_data()

    df = clean_data(df)
    df = encode_target(df)
    df, encoders = encode_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "feature_names": X.columns.tolist(),
        "encoders": encoders,
        "scaler": scaler,
        "df_clean": df,
    }


def apply_oversampling(X_train, y_train, random_state=RANDOM_STATE):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def apply_downsampling(X_train, y_train, random_state=RANDOM_STATE):
    undersampler = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_class_distribution(y):
    if hasattr(y, "value_counts"):
        counts = y.value_counts().to_dict()
    else:
        unique, counts_arr = np.unique(y, return_counts=True)
        counts = dict(zip(unique, counts_arr))
    return {str(k): int(v) for k, v in counts.items()}
