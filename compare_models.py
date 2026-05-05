import argparse
import os
import json
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not found — skipping plots. pip install matplotlib")

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        cohen_kappa_score,
        precision_recall_fscore_support
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not found — some metrics skipped. pip install scikit-learn")



EXPECTED_COLS = ["text", "is_inappropriate", "is_relevant", "decision"]
BINARY_COLS   = ["is_inappropriate", "is_relevant"]
DECISION_COL  = "decision"
TEXT_COL      = "text"
COLORS        = ["#4C72B0", "#DD8452"]   # blue / orange



def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] {path} is missing expected columns: {missing}")
        print(f"Found columns: {df.columns.tolist()}")

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if DECISION_COL in df.columns:
        df[DECISION_COL] = df[DECISION_COL].astype(str).str.strip().str.upper()

    return df


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"  Saved -> {path}")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)



def align_dataframes(df1, df2, name1, name2):
    if TEXT_COL in df1.columns and TEXT_COL in df2.columns:
        merged = pd.merge(df1, df2, on=TEXT_COL, suffixes=(f"_{name1}", f"_{name2}"), how="inner")
        n = len(merged)

        print(f"\n[INFO] Matched {n} posts by text "
              f"({len(df1)} in {name1}, {len(df2)} in {name2})")
        
        if n == 0:
            print("[WARN] No matching texts — falling back to row alignment.")
            return positional_align(df1, df2, name1, name2)
        return merged
    return positional_align(df1, df2, name1, name2)


def positional_align(df1, df2, name1, name2):
    min_len = min(len(df1), len(df2))
    if len(df1) != len(df2):
        print(f"[WARN] Files have different lengths ({len(df1)} vs {len(df2)}). "
              f"Truncating to {min_len} rows.")
        
    df1 = df1.iloc[:min_len].reset_index(drop=True)

    df2 = df2.iloc[:min_len].reset_index(drop=True)

    df1.columns = [f"{c}_{name1}" if c != TEXT_COL else c for c in df1.columns]
    df2.columns = [f"{c}_{name2}" if c != TEXT_COL else c for c in df2.columns]

    return pd.concat([df1, df2], axis=1)




def binary_agreement(merged, col, name1, name2):
    c1, c2 = f"{col}_{name1}", f"{col}_{name2}"
    if c1 not in merged.columns or c2 not in merged.columns:
        return {}

    a, b   = merged[c1], merged[c2]
    agree  = (a == b).sum()
    total  = len(merged)

    stats = {
        "total_posts":                  total,
        "agreements":                   int(agree),
        "disagreements":                int(total - agree),
        "agreement_pct":                round(agree / total * 100, 2),
        f"{name1}_positive_rate_%":     round(a.mean() * 100, 2),
        f"{name2}_positive_rate_%":     round(b.mean() * 100, 2),
    }

    if HAS_SKLEARN:
        stats["cohen_kappa"] = round(cohen_kappa_score(a, b), 4)
        p, r, f1, _ = precision_recall_fscore_support(
            a, b, average="binary", zero_division=0)
        stats[f"precision_{name2}_vs_{name1}"] = round(p, 4)
        stats[f"recall_{name2}_vs_{name1}"]    = round(r, 4)
        stats[f"f1_{name2}_vs_{name1}"]        = round(f1, 4)

    return stats


def decision_agreement(merged, name1, name2):
    c1, c2 = f"{DECISION_COL}_{name1}", f"{DECISION_COL}_{name2}"
    if c1 not in merged.columns or c2 not in merged.columns:
        return {}

    a, b  = merged[c1], merged[c2]
    total = len(merged)
    agree = (a == b).sum()

    all_labels = sorted(set(a.unique()) | set(b.unique()))
    stats = {
        "total_posts":        total,
        "agreements":         int(agree),
        "disagreements":      int(total - agree),
        "agreement_pct":      round(agree / total * 100, 2),
        "all_decision_labels": all_labels,
    }

    for label in all_labels:
        stats[f"{name1}_{label}_count"] = int((a == label).sum())
        stats[f"{name2}_{label}_count"] = int((b == label).sum())

    if HAS_SKLEARN:
        stats["cohen_kappa"]           = round(cohen_kappa_score(a, b), 4)
        stats["classification_report"] = classification_report(
            a, b, labels=all_labels, zero_division=0)

    return stats



def export_disagreements(merged, col, name1, name2, out_dir):
    c1, c2 = f"{col}_{name1}", f"{col}_{name2}"
    if c1 not in merged.columns or c2 not in merged.columns:
        return
    mask = merged[c1] != merged[c2]
    rows = merged[mask].copy()
    if len(rows) == 0:
        print(f"  [{col}] Perfect agreement — no disagreements!")
        return
    rows["disagreement_col"] = col
    path = os.path.join(out_dir, f"disagreements_{col}.csv")
    save_csv(rows, path)
    print(f"  [{col}] {len(rows)} disagreements exported.")




def plot_agreement_bar(stats, col, name1, name2, out_dir):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [stats.get("agreements", 0), stats.get("disagreements", 0)]
    bars   = ax.bar(["Agree", "Disagree"], values,
                    color=["#2ecc71", "#e74c3c"], width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(val),
                ha="center", va="bottom", fontsize=11)
    ax.set_title(f"Agreement on '{col}'\n{name1}  vs  {name2}", fontsize=13)
    ax.set_ylabel("Number of posts")
    ax.set_ylim(0, max(values) * 1.2 + 1)
    plt.tight_layout()
    path = os.path.join(out_dir, f"agreement_{col}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


def plot_decision_distribution(merged, labels, name1, name2, out_dir):
    if not HAS_MATPLOTLIB:
        return
    c1, c2 = f"{DECISION_COL}_{name1}", f"{DECISION_COL}_{name2}"
    if c1 not in merged.columns or c2 not in merged.columns:
        return
    counts1 = [(merged[c1] == l).sum() for l in labels]
    counts2 = [(merged[c2] == l).sum() for l in labels]
    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 5))
    for bars, counts, color, name in [
        (ax.bar(x - w/2, counts1, w, color=COLORS[0]), counts1, COLORS[0], name1),
        (ax.bar(x + w/2, counts2, w, color=COLORS[1]), counts2, COLORS[1], name2),
    ]:
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2, str(int(val)),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"Decision Distribution: {name1}  vs  {name2}", fontsize=13)
    ax.legend([name1, name2])
    plt.tight_layout()
    path = os.path.join(out_dir, "decision_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


def plot_confusion_matrix(merged, col, name1, name2, labels, out_dir):
    if not (HAS_MATPLOTLIB and HAS_SKLEARN):
        return
    c1, c2 = f"{col}_{name1}", f"{col}_{name2}"
    if c1 not in merged.columns or c2 not in merged.columns:
        return
    cm = confusion_matrix(merged[c1], merged[c2], labels=labels)
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels], rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([str(l) for l in labels], fontsize=9)
    ax.set_xlabel(f"Predicted ({name2})", fontsize=11)
    ax.set_ylabel(f"Reference ({name1})", fontsize=11)
    ax.set_title(f"Confusion Matrix — {col}\n{name1}  vs  {name2}", fontsize=12)
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{col}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


def plot_positive_rates(all_stats, name1, name2, out_dir):
    if not HAS_MATPLOTLIB:
        return
    rows = [(col, s[f"{name1}_positive_rate_%"], s[f"{name2}_positive_rate_%"])
            for col, s in all_stats.items()
            if f"{name1}_positive_rate_%" in s]
    if not rows:
        return
    cols_  = [r[0] for r in rows]
    vals1  = [r[1] for r in rows]
    vals2  = [r[2] for r in rows]
    y, h   = np.arange(len(cols_)), 0.35
    fig, ax = plt.subplots(figsize=(8, max(3, len(cols_) * 1.2)))
    ax.barh(y - h/2, vals1, h, label=name1, color=COLORS[0])
    ax.barh(y + h/2, vals2, h, label=name2, color=COLORS[1])
    ax.set_yticks(y)
    ax.set_yticklabels(cols_)
    ax.set_xlabel("Positive rate (%)")
    ax.set_title(f"Positive Rates: {name1}  vs  {name2}", fontsize=13)
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.8)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "positive_rates.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")




def print_summary(all_stats, name1, name2):
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY:  {name1}  vs  {name2}")
    print(f"{'='*60}")
    for col, stats in all_stats.items():
        print(f"\n{'─'*60}\n  Column: {col}\n{'─'*60}")
        for k, v in stats.items():
            if k == "classification_report":
                print(f"\n  Classification Report "
                      f"({name2} treating {name1} as reference):\n{v}")
            else:
                print(f"  {k}: {v}")


def save_summary_json(all_stats, out_dir, name1, name2):
    clean = {col: {k: v for k, v in s.items() if k != "classification_report"}
             for col, s in all_stats.items()}
    path = os.path.join(out_dir, "summary_stats.json")
    with open(path, "w") as f:
        json.dump({"model1": name1, "model2": name2, "stats": clean}, f, indent=2)
    print(f"  Saved -> {path}")


def save_summary_txt(all_stats, out_dir, name1, name2):
    path = os.path.join(out_dir, "summary_report.txt")
    with open(path, "w") as f:
        f.write(f"COMPARISON REPORT\nModel 1: {name1}\nModel 2: {name2}\n\n")
        for col, stats in all_stats.items():
            f.write(f"{'─'*60}\nColumn: {col}\n{'─'*60}\n")
            for k, v in stats.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"  Saved -> {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare two moderation CSV files and generate statistics.")
    parser.add_argument("--file1",    required=True, help="Path to first CSV")
    parser.add_argument("--file2",    required=True, help="Path to second CSV")
    parser.add_argument("--name1",    default="Model_1", help="Label for model 1")
    parser.add_argument("--name2",    default="Model_2", help="Label for model 2")
    parser.add_argument("--output",   default="./comparison_results",
                        help="Output directory (default: ./comparison_results)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating charts")
    args = parser.parse_args()

    make_plots = (not args.no_plots) and HAS_MATPLOTLIB

    print(f"\nLoading {args.file1} ...")
    df1 = load_csv(args.file1)
    print(f"  Rows: {len(df1)}  |  Columns: {df1.columns.tolist()}")

    print(f"\nLoading {args.file2} ...")
    df2 = load_csv(args.file2)
    print(f"  Rows: {len(df2)}  |  Columns: {df2.columns.tolist()}")

    merged = align_dataframes(df1, df2, args.name1, args.name2)
    ensure_dir(args.output)
    save_csv(merged, os.path.join(args.output, "merged_comparison.csv"))

    all_stats = {}

  
    print("\n── Binary column analysis ──────────────────────────────")
    for col in BINARY_COLS:
        stats = binary_agreement(merged, col, args.name1, args.name2)
        if stats:
            all_stats[col] = stats
            export_disagreements(merged, col, args.name1, args.name2, args.output)
            if make_plots:
                plot_agreement_bar(stats, col, args.name1, args.name2, args.output)
                plot_confusion_matrix(merged, col, args.name1, args.name2,
                                      labels=[0, 1], out_dir=args.output)

    if make_plots and all_stats:
        plot_positive_rates(all_stats, args.name1, args.name2, args.output)

    # Decision column
    print("\n── Decision column analysis ────────────────────────────")
    d_stats = decision_agreement(merged, args.name1, args.name2)
    if d_stats:
        all_stats[DECISION_COL] = d_stats
        export_disagreements(merged, DECISION_COL,
                             args.name1, args.name2, args.output)
        if make_plots:
            labels = d_stats.get("all_decision_labels", [])
            plot_decision_distribution(merged, labels,
                                       args.name1, args.name2, args.output)
            plot_confusion_matrix(merged, DECISION_COL, args.name1, args.name2,
                                  labels=labels, out_dir=args.output)

    print_summary(all_stats, args.name1, args.name2)

    print("\n── Saving results ──────────────────────────────────────")
    save_summary_json(all_stats, args.output, args.name1, args.name2)
    save_summary_txt(all_stats, args.output, args.name1, args.name2)

    print(f"\nDone! All results saved to: {args.output}/\n")


if __name__ == "__main__":
    main()