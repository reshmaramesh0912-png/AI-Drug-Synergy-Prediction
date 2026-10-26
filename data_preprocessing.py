#AI_DRUG_COMBINATION
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("summary_table_v1.4.csv")
print(df.head())
print(df.columns)
print(df.shape)
#handling missing values
print(df.isnull().sum())
df = df.dropna(subset=["drug_col"])
print(df.shape)
print(df.isnull().sum())
#dropping duplicates
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
#knowing the data types of each column
print(df.info())
#print(df.describe())
#EDA
all_drugs = pd.concat([df["drug_row"], df["drug_col"]])
print("Unique drugs:", all_drugs.nunique())
print(df["cell_line_name"].nunique())
print(all_drugs.value_counts().head(10))
#Distribution of synergy scores
plt.figure(figsize=(8,5))
plt.hist(df["synergy_zip"], bins=50)
plt.xlabel("ZIP Synergy Score")
plt.ylabel("Frequency")
plt.title("Distribution of ZIP Synergy Scores")
plt.tight_layout()
plt.savefig("synergy_distribution.png")
plt.close()
print("Histogram completed")
#print(df["synergy_zip"].describe())
#correlation heatmap
cols = [
    "css",
    "synergy_zip",
    "synergy_bliss",
    "synergy_loewe",
    "synergy_hsa",
    "ic50_row",
    "ic50_col",
    "ri_row",
    "ri_col",
    "css_row",
    "css_col"
]
sample_df = df.sample(10000, random_state=42)
corr = sample_df[cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True,fmt=".2f",cmap="coolwarm",linewidth=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png",dpi=300)
plt.close()
print(df["cell_line_name"].value_counts().head(10))
df["synergy"] = (df["synergy_zip"] > 10).astype(int)
print(df["synergy"].value_counts())
df.to_csv("cleaned_drugcomb.csv", index=False)
print("Data cleaning completed.")