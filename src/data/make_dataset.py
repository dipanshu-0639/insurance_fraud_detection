import pandas as pd
import numpy as np
import os


def make_dataset(input_path, output_path):

    print("Reading raw dataset...")
    df = pd.read_csv(input_path)

    df.replace('?', np.nan, inplace=True)


    df["collision_type"].fillna(  df["collision_type"].mode()[0], inplace=True)

    df["property_damage"].fillna(df["property_damage"].mode()[0],inplace=True)

    df["police_report_available"].fillna(df["police_report_available"].mode()[0],inplace=True)

    print("Missing values handled")

    to_drop = [
        'policy_number',
        'policy_bind_date',
        'policy_state',
        'insured_zip',
        'incident_location',
        'incident_date',
        'incident_state',
        'incident_city',
        'insured_hobbies',
        'auto_make',
        'auto_model',
        'auto_year',
        '_c39'
    ]

    df.drop(columns=to_drop, inplace=True, errors="ignore")

 
    df.drop(columns=['age', 'total_claim_amount'],inplace=True,errors="ignore")

    print("Unnecessary columns dropped ")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved at: {output_path} ")

if __name__ == "__main__":

    make_dataset(input_path="data/raw/insurance_claims.csv",output_path="data/processed/cleaned_insurance_claims.csv")