import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def load_data(path):

    df = pd.read_csv(path)

    X = df.drop("fraud_reported", axis=1)
    y = df["fraud_reported"]

    return train_test_split( X, y, test_size=0.2, random_state=42)

def get_models():

    return {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }


search_spaces = {

    "Decision Tree": {
        "max_depth": hp.choice("max_depth", range(3, 15)),
        "min_samples_split": hp.uniform("min_samples_split", 0.1, 1.0)
    },

    "Random Forest": {
        "n_estimators": hp.choice("n_estimators", range(50, 200)),
        "max_depth": hp.choice("max_depth", range(5, 20))
    },

    "Extra Trees": {
        "n_estimators": hp.choice("n_estimators", range(50, 200)),
        "max_depth": hp.choice("max_depth", range(5, 20))
    },

    "XGBoost": {
        "n_estimators": hp.choice("n_estimators", range(50, 200)),
        "max_depth": hp.choice("max_depth", range(3, 10)),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3)
    }
}


def objective(params, model_name, X_train, X_test, y_train, y_test):

    model = get_models()[model_name]

    model.set_params(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    recall = recall_score(y_test, preds)

    return {
        "loss": -recall,
        "status": STATUS_OK
    }


def find_best_model(data_path):

    X_train, X_test, y_train, y_test = load_data(data_path)

    best_models = {}

    for model_name in get_models().keys():

        print(f"\n Tuning {model_name}...")

        trials = Trials()

        best_params = fmin(
            fn=lambda params: objective(
                params,
                model_name,
                X_train,
                X_test,
                y_train,
                y_test
            ),
            space=search_spaces[model_name],
            algo=tpe.suggest,
            max_evals=25,
            trials=trials
        )

        model = get_models()[model_name]
        model.set_params(**best_params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        best_models[model_name] = {
            "model": model,
            "recall": recall,
            "f1": f1
        }

        print(
            f"{model_name} → Recall: {recall:.3f}, F1: {f1:.3f}"
        )


    best_name = max(
        best_models,
        key=lambda x: best_models[x]["recall"]
    )

    best_model = best_models[best_name]["model"]

    print(f"\n Best Model: {best_name}")

   
    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model,"models/best_fraud_model.pkl")

    print("Best model saved ")

if __name__ == "__main__":

    find_best_model( "data/processed/label_encoded_data.csv")