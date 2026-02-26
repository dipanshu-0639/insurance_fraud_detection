import pandas as pd
import os
import joblib

from sklearn.preprocessing import LabelEncoder


def label_encode_data(input_path, output_path, encoder_path):

    print("Reading cleaned dataset...")
    df = pd.read_csv(input_path)
    

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
  

    print(f"Categorical columns found: {len(categorical_cols)}")

    encoders = {}

    for col in categorical_cols:

        print(f"Encoding: {col}")

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

    print("Label encoding completed ")

    os.makedirs(encoder_path, exist_ok=True)

    for col, encoder in encoders.items():

        joblib.dump(encoder,f"{encoder_path}/{col}_encoder.pkl")

    print("Encoders saved successfully ")


    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    

    df.to_csv(output_path, index=False)

    print(f"Encoded dataset saved at: {output_path} ")



if __name__ == "__main__":

    label_encode_data(
        input_path="data/processed/cleaned_insurance_claims.csv",
        output_path="data/processed/label_encoded_data.csv",
        encoder_path="models/encoders"
    )