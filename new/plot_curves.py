import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np

root = Path.cwd()
num = "6002"
out_csv = root / f"all_losses{num}.csv"


exp_effects = {
    f"exp_{num}": "x",
}
def detect_series(m):
    m = m.lower()
    if re.search(r"s[\W_]?21", m):
        return "s21"
    if re.search(r"s[\W_]?22", m):
        return "s22"
    if re.search(r"s[\W_]?2", m):
        return "s2"
    return "other"

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
last_batch_records = []

for exp_id, effect in exp_effects.items():
    exp_dir = root / exp_id
    for subdir in exp_dir.iterdir():
        if not subdir.is_dir():
            continue
        model = subdir.name
        print(subdir.name)
        training = subdir.name.split("_")[0] if "_" in subdir.name else "unknown"
        csvs = list(subdir.glob("losses-*.csv"))
        if not csvs:
            continue
        for f in csvs:
            run_id = f.stem.split("-")[-1] 
            df = pd.read_csv(f)
            
            last_batch = df.iloc[-1]
            last_batch_records.append({
                "exp_id": exp_id,
                "model": model,
                "run_id": run_id,
                "novel_accuracy": last_batch["Novel_Accuracy"],
                "sim_i_loss": last_batch["sim_i_loss"],
                "sim_r_loss": last_batch["sim_r_loss"],
                "novel_i_loss": last_batch["novel_sim_i_loss"],
                "novel_r_loss": last_batch["novel_sim_r_loss"],
            })
            
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
            combined["nsl"] = "nsl" in model
            combined["run_id"] = run_id
            combined["arch"] = extract_arch(model)
            records.append(combined)


if not records:
    raise SystemExit("No losses-*.csv files found")

if last_batch_records:
    last_df = pd.DataFrame(last_batch_records)
    summary = (
            last_df
            .groupby(["exp_id", "model"])[["novel_accuracy", "novel_i_loss", "novel_r_loss", "sim_r_loss", "sim_i_loss"]]
            .agg(["mean", "std", "count"])
        )
    print("\n" + "="*80)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    print("LAST BATCH NOVEL_TASK_ACCURACY SUMMARY")
    for (exp_id, model), row in summary.iterrows():
        print(
            f"{model}, {int(row['novel_accuracy_count'])}, "
            f"{row['novel_accuracy_mean']:.4f}, {row['novel_accuracy_std']:.5f}, "
            f"{row['sim_i_loss_mean']:.4f}, {row['sim_r_loss_mean']:.4f},"
            f"{row['novel_i_loss_mean']:.4f}, {row['novel_r_loss_mean']:.4f}"
        )

    def top3_stats(group):
        top3 = group.nlargest(5, 'novel_accuracy')
        return pd.Series({
            'novel_accuracy_mean': top3['novel_accuracy'].mean(),
            'novel_accuracy_std': top3['novel_accuracy'].std(),
            'novel_i_loss_mean': top3['novel_i_loss'].mean(),
            'novel_r_loss_mean': top3['novel_r_loss'].mean(),
            'sim_r_loss_mean': top3['sim_r_loss'].mean(),
            'sim_i_loss_mean': top3['sim_i_loss'].mean(),
            'count': len(top3)
        })

    summary = last_df.groupby(["exp_id", "model"]).apply(top3_stats)
    
    print("\n" + "="*80)
    print("LAST BATCH NOVEL_TASK_ACCURACY SUMMARY (TOP 5)")
    for (exp_id, model), row in summary.iterrows():
        print(
            f"{model}, {int(row['count'])}, "
            f"{row['novel_accuracy_mean']:.4f}, {row['novel_accuracy_std']:.5f}, "
            f"{row['sim_i_loss_mean']:.4f}, {row['sim_r_loss_mean']:.4f},"
            f"{row['novel_i_loss_mean']:.4f}, {row['novel_r_loss_mean']:.4f}"
        )

big_df = pd.concat(records, ignore_index=True)
print(sorted(big_df["model"].unique()))
big_df["series"] = big_df["model"].apply(detect_series)
big_df["modeltype"] = big_df["model"].apply(lambda m: "shared" if "shared" in m else "single" if "single" in m else "split")
big_df.to_csv(out_csv, index=False)
print(f"Saved combined CSV: {out_csv}")

if True:
    dfa = big_df
    for arch in ['transformer32']: 
        dfb = dfa[dfa["arch"]==arch]
        dfb = dfa
        for series in ["s21","s22", "s2"]:
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

            loss_metrics = [m for m in ["Loss_mean", "Novel_Loss_mean", "Novel_Task_Loss_mean",] if m in dfc.columns] #  "sim_r_loss_mean", "sim_i_loss_mean"
            styles = ["-", "--", ":"]
            if loss_metrics:
                plt.figure(figsize=[10,8])

                configs = sorted(dfc.groupby(["pad","swap","oracle","model"]).groups.keys())
                colors = plt.cm.tab10.colors
                color_map = {cfg: colors[i % len(colors)] for i, cfg in enumerate(configs)}

                for cfg in configs:
                    pad, swap, oracle, model = cfg
                    color = color_map[cfg]
                    label = f"{model}, pad={pad}, swap={swap}, oracle={oracle}"

                    g = (
                        dfc[dfc["pad"].eq(pad) & dfc["swap"].eq(swap) &
                            dfc["oracle"].eq(oracle) & dfc["model"].eq(model)]
                        .groupby("Batch")[loss_metrics].mean()
                        .reset_index()
                        .sort_values("Batch")
                    )

                    for lm, style in zip(loss_metrics, styles):
                        print(lm, len(g[lm]), g[lm].head())
                        plt.plot(g["Batch"], g[lm], linestyle=style, color=color, linewidth=1.5, label=f"{label} {lm}")

                    plt.plot([], [], color=color, label=label)

                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.ylim([0.0, 1.2])
                plt.title(f"Loss Metrics Across Runs ({series}-{arch})")
                plt.legend(fontsize=7)
                plt.tight_layout()
                plt.savefig(root / f"exp_{num}/losses_{series}_{arch}_{num}.png")
                plt.close()
            acc_metrics = [m for m in ["Novel_Accuracy_mean","Accuracy_mean","Novel_Task_Accuracy_mean"] if m in dfc.columns]
            if acc_metrics:
                plt.figure(figsize=[10,10])
                
                configs = sorted(dfc.groupby(["pad","swap","oracle","model"]).groups.keys())
                colors = plt.cm.tab10.colors
                color_map = {cfg: colors[i % len(colors)] for i, cfg in enumerate(configs)}
                
                for cfg in configs:
                    pad, swap, oracle, model = cfg
                    color = color_map[cfg]
                    
                    subset = dfc[dfc["pad"].eq(pad) & dfc["swap"].eq(swap) & 
                                 dfc["oracle"].eq(oracle) & dfc["model"].eq(model)]
                    
                    for run_id, g in subset.groupby("run_id"):
                        g_sorted = g.sort_values("Batch")
                        plt.plot(g_sorted["Batch"], g_sorted["Novel_Accuracy_mean"],  color=color, alpha=0.6, linewidth=0.8)
                
                base = (
                    dfc.groupby(["pad","swap","oracle","Batch","model"])[acc_metrics]
                    .agg(["mean","std"])
                    .reset_index()
                    .groupby(["pad","swap","oracle","model"])
                )
                for (pad, swap, oracle, model), g in base:
                    label = f"{model}, pad={pad}, swap={swap}, oracle={oracle}"
                    color = color_map[(pad, swap, oracle, model)]
                    first = acc_metrics[0]
                    f_mean = g[(first, "mean")]
                    f_std  = g[(first, "std")]
                    plt.plot(g["Batch"], f_mean, label=label, color=color)
                    plt.fill_between(g["Batch"], f_mean - f_std, f_mean + f_std, alpha=0.2, color=color)

                    for am, style in zip(acc_metrics[1:], styles[1:]):
                        m_mean = g[(am, "mean")]
                        m_std  = g[(am, "std")]
                        plt.plot(g["Batch"], m_mean, linestyle=style, linewidth=1, color=color)
                        plt.fill_between(g["Batch"], m_mean - m_std, m_mean + m_std, alpha=0.2, color=color)

                plt.xlabel("Batch")
                plt.ylabel("Accuracy")
                plt.ylim([0.25, 1.0])
                plt.title(f"Accuracy ({series}-{arch})")
                plt.legend(fontsize=7)
                plt.tight_layout()
                plt.savefig(root / f"exp_{num}/accuracy_all_{series}_{arch}_{num}.png")
                plt.close()