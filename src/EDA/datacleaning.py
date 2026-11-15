import pandas as pd

df = pd.read_csv("data/LifeInsurancePolicyData.csv", encoding="utf-8-sig")
print("Rows:", len(df))
print("Duplicate rows (kept):", df.duplicated().sum())

# Fix Quarterly typo
mask = df["PAYMENT MODE"] == "Quaterly"
df.loc[mask, "PAYMENT MODE"] = "Quarterly"
print("Fixed 'Quaterly' -> 'Quarterly':", mask.sum())

mask = df["Premium"] <= 0
df.loc[mask, "Premium"] = None
print("Premium <= 0 set to missing:", mask.sum())

mask = df["SUBSTANDARD RISK"] < 0
df.loc[mask, "SUBSTANDARD RISK"] = 0
print("Negative SUBSTANDARD RISK set to 0:", mask.sum())

# drop CHANNEL columns
df = df.drop(columns = df.columns[df.columns.str.startswith('CHANNEL')])

df['is_lapse'] = (df['POLICY STATUS'] == "Lapse").astype(int)

print("\nMissing values after cleaning:")
print(df.isna().sum()[df.isna().sum() > 0])

df.to_csv("data/CleanedLifeInsurance.csv", index=False)
print("\nWrote data/CleanedLifeInsurance.csv")