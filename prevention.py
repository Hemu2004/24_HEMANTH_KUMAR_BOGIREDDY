import pandas as pd

df_precaution = pd.read_csv("Disease precaution.csv")

def get_precautions(disease):
    row = df_precaution[df_precaution["Disease"] == disease]

    if row.empty:
        return []

    precautions = row.iloc[0, 1:].dropna().tolist()
    return precautions
