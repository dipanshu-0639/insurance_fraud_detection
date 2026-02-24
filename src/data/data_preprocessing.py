import pandas as pd
import numpy as np
import os


def preprocess_data(input_path, output_path):

    print("Reading cleaned dataset...")
    df = pd.read_csv(input_path)

    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1,'N': 0})

    print("Target column encoded ")

    target_col = "fraud_reported"

    y = df[target_col]
    X = df.drop(columns=[target_col])


    X_encoded = pd.get_dummies( X, drop_first=True)

    print("Categorical encoding completed ")

    final_df = pd.concat([X_encoded, y],axis=1)

    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    final_df.to_csv(output_path,index=False)

    print(f"Encoded dataset saved at: {output_path} ")

if __name__ == "__main__":

    preprocess_data(input_path="data/processed/cleaned_insurance_claims.csv",output_path="data/processed/encoded_insurance_claims.csv")