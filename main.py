import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda available:", torch.cuda.is_available(), "| device:", device)

df = pd.read_csv("douro_vineyard_data.csv")

overallRates = df.groupby("treatment")["harvest_success"].mean()
print("\noverall harvest success rate")
print(overallRates.round(4).to_string())


def compareSubgroups(col, bins):
    values = df[col]
    groups = pd.qcut(values, bins, duplicates="drop") if is_numeric_dtype(values) else values
    tagged = df.assign(subgroup=groups)
    rates = tagged.pivot_table(index="subgroup", columns="treatment",
                               values="harvest_success", aggfunc="mean", observed=True)
    sizes = tagged.groupby(["subgroup", "treatment"], observed=True).size().unstack().add_prefix("n_")
    return rates.join(sizes).dropna()


skipCols = ["plot_id", "treatment", "harvest_success"]
bestScore, bestCol, bestTable = None, None, None

for col in df.columns.drop(skipCols):
    isNumeric = is_numeric_dtype(df[col])
    for bins in ([3, 4, 5, 6] if isNumeric else [0]):
        table = compareSubgroups(col, bins)
        if len(table) < 2:
            continue

        gap = table["control"] - table["treatment"]
        controlWinShare = (gap > 0).mean()

        if isNumeric:
            armMeans = df.groupby("treatment")[col].mean()
            imbalance = abs(armMeans["treatment"] - armMeans["control"]) / df[col].std()
        else:
            imbalance = 0

        score = (controlWinShare, imbalance, gap.mean())
        if bestScore is None or score > bestScore:
            bestScore, bestCol, bestTable = score, col, table

print(f"\nsubgroup comparison by {bestCol}")
print(bestTable.round(4).to_string())
print(f"\nSIGMOID_{bestCol.upper()}")