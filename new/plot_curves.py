import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

root = Path.cwd()
out_csv = root / "all_losses.csv"

exp_effects = {
    #"exp_555": "pad 8, loss=D+O, size swaps",
    #"exp_556": "pad 5, loss=D+O, size swaps",
    #"exp_557": "pad 8, loss=D, size swaps",
    #"exp_558": "pad 5, loss=D, size swaps",
    #"exp_567": "pad 8, loss=D",
    #"exp_568": "pad 5, loss=D",
    "exp_600": "x",
}


def contrast(df, var):
    others = [c for c in ["pad", "swap", "oracle"] if c != var]
    diffs = []
    for _, g in df.groupby(others):
        if g[var].nunique() == 2:
            v_true = g.loc[g[var], "Novel_Accuracy_mean"].mean()
            v_false = g.loc[~g[var], "Novel_Accuracy_mean"].mean()
            diffs.append(v_true - v_false)
    return pd.Series(diffs).mean(), pd.Series(diffs).std()

def extract_arch(model):
    match = re.search(r"(mlp|lstm\d+|transformer\d+)", model)
    return match.group(1) if match else "unknown"

records = []
for exp_id, effect in exp_effects.items():
    exp_dir = root / exp_id
    for subdir in exp_dir.iterdir():
        if not subdir.is_dir():
            continue
        model = subdir.name
        training = subdir.name.split("_")[0] if "_" in subdir.name else "unknown"
        csvs = list(subdir.glob("losses-*.csv"))
        if not csvs:
            continue
        for f in csvs:
            run_id = f.stem.split("-")[-1] 
            df = pd.read_csv(f)
            combined = df.groupby("Batch").agg(["mean", "std"])
            combined.columns = [f"{a}_{b}" for a, b in combined.columns]
            combined = combined.reset_index()
            combined["exp_id"] = exp_id
            combined["effect"] = effect
            combined["model"] = model
            combined["training"] = training
            combined["pad"] = "pad" in model
            combined["swap"] = "swap" in model
            combined["oracle"] = "oracle" in model
            combined["run_id"] = run_id
            combined["arch"] = extract_arch(model)
            records.append(combined)


if not records:
    raise SystemExit("No losses-*.csv files found")

big_df = pd.concat(records, ignore_index=True)
big_df["series"] = big_df["model"].apply(lambda m: "s21" if "s21" in m else ("s22" if "s22" in m else "other"))
big_df["modeltype"] = big_df["model"].apply(lambda m: "sym" if "sym" in m else "full")
big_df.to_csv(out_csv, index=False)
print(f"Saved combined CSV: {out_csv}")

for modelType in ['sym', 'full']:  
    dfa = big_df[big_df["modeltype"]==modelType]
    for arch in ['mlp', 'lstm32', 'transformer32']: 
        dfb = dfa[dfa["arch"]==arch]
        for series in ["s21","s22"]:
            dfc = dfb[dfb["series"]==series]
            if dfc.empty:
                continue

            per_run = dfc.groupby(["model","pad","swap","oracle","run_id"], as_index=False)["Novel_Accuracy_mean"].mean()
            summary = dfc.groupby(["pad","swap","oracle"], as_index=False)["Novel_Accuracy_mean"].mean()
            summary_std = dfc.groupby(["pad","swap","oracle"], as_index=False)["Novel_Accuracy_mean"].std()

            print(f"\n[{series}] Means:\n", summary)
            print(f"\n[{series}] Stds:\n", summary_std)

            for var in ["pad","swap","oracle"]:
                mean_diff, std_diff = contrast(per_run, var)
                print(f"[{series}] {var}: mean change={mean_diff:.4f} ({std_diff:.4f})")

            subset = per_run[per_run["oracle"]]
            for var in ["pad","swap"]:
                mean_diff, std_diff = contrast(subset, var)
                print(f"[{series}] {var} (oracle=True): mean change={mean_diff:.4f} ({std_diff:.4f})")

            plt.figure(figsize=[10,8])
            for (pad, swap, oracle, model), g in (
                dfc.groupby(["pad","swap","oracle","Batch","model"])["Novel_Accuracy_mean"]
                .agg(["mean","std"])
                .reset_index()
                .groupby(["pad","swap","oracle","model"])
            ):
                label = f"{model}, pad={pad}, swap={swap}, oracle={oracle}"
                plt.plot(g["Batch"], g["mean"], label=label)
                plt.fill_between(g["Batch"], g["mean"]-g["std"], g["mean"]+g["std"], alpha=0.2)

            plt.xlabel("Batch")
            plt.ylabel("Accuracy")
            plt.ylim([0.7,1.0])
            plt.title(f"Accuracy Across Runs ({series}-{arch})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(root / f"accuracy_{series}_{modelType}_{arch}_604d.png")
            plt.close()