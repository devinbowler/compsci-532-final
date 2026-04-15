import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def mean_std(df, group_cols, value_col="time"):
    """Return a df with mean and std of value_col, grouped by group_cols."""
    return (
        df.groupby(group_cols)[value_col]
        .agg(mean="mean", std="std")
        .reset_index()
    )

def plot_df_vs_sql_per_query(df, out_dir):
    """
    One figure per query: mean execution time vs partitions for each method,
    with ±1 std error bands. Separate subplots for cached / uncached.
    """
    queries = sorted(df["query"].unique())

    for q in queries:
        subset = df[df["query"] == q]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        fig.suptitle(f"{q.upper()}  —  DataFrame vs SQL", fontsize=13)

        for ax, cache in zip(axes, [True, False]):
            grp = mean_std(
                subset[subset["cache"] == cache],
                ["partitions", "method"],
            )
            for method, color in [("dataframe", "#2196F3"), ("sql", "#FF5722")]:
                m = grp[grp["method"] == method]
                ax.plot(m["partitions"], m["mean"], marker="o",
                        label=method, color=color)
                ax.fill_between(
                    m["partitions"],
                    m["mean"] - m["std"],
                    m["mean"] + m["std"],
                    alpha=0.18, color=color,
                )
            ax.set_title(f"cache={'on' if cache else 'off'}")
            ax.set_xlabel("Partitions")
            ax.set_ylabel("Time (s)")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend()
            ax.grid(alpha=0.4)

        fig.tight_layout()
        save(fig, os.path.join(out_dir, f"{q}_df_vs_sql.png"))


def plot_speedup_ratio(df, out_dir):
    """
    SQL / DataFrame speedup ratio vs partitions, one line per query.
    Ratio > 1 means SQL is faster; < 1 means DataFrame is faster.
    Separate subplots for cached / uncached.
    """
    base = mean_std(df, ["query", "partitions", "cache", "method"])
    pivot = base.pivot_table(
        index=["query", "partitions", "cache"],
        columns="method",
        values="mean",
    ).reset_index()
    pivot["speedup"] = pivot["sql"] / pivot["dataframe"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.suptitle("SQL / DataFrame Speedup Ratio  (>1 = SQL faster)", fontsize=13)

    queries = sorted(pivot["query"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {q: cmap(i) for i, q in enumerate(queries)}

    for ax, cache in zip(axes, [True, False]):
        grp = pivot[pivot["cache"] == cache]
        for q in queries:
            m = grp[grp["query"] == q]
            ax.plot(m["partitions"], m["speedup"], marker="o",
                    label=q, color=colors[q])
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.2, label="parity")
        ax.set_title(f"cache={'on' if cache else 'off'}")
        ax.set_xlabel("Partitions")
        ax.set_ylabel("Speedup ratio")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "speedup_ratio.png"))


def plot_cache_effect(df, out_dir):
    """
    Grouped bar chart: for each (query, method) pair, show mean time
    with cache on vs off. Error bars show ±1 std across trials.
    """
    grp = mean_std(df, ["query", "method", "cache"])

    queries = sorted(grp["query"].unique())
    methods = ["dataframe", "sql"]
    cache_vals = [True, False]
    cache_labels = {True: "cache on", False: "cache off"}
    cache_colors = {True: "#4CAF50", False: "#F44336"}

    for method in methods:
        sub = grp[grp["method"] == method]
        x = np.arange(len(queries))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 4))
        for i, cache in enumerate(cache_vals):
            m = sub[sub["cache"] == cache].set_index("query").reindex(queries)
            ax.bar(
                x + (i - 0.5) * width,
                m["mean"],
                width,
                yerr=m["std"],
                capsize=4,
                label=cache_labels[cache],
                color=cache_colors[cache],
                alpha=0.85,
            )

        ax.set_title(f"Cache Effect  —  {method.capitalize()}")
        ax.set_xlabel("Query")
        ax.set_ylabel("Mean time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels([q.upper() for q in queries])
        ax.legend()
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()
        save(fig, os.path.join(out_dir, f"cache_effect_{method}.png"))


def plot_partition_heatmaps(df, out_dir):
    """
    Two heatmaps side by side (DataFrame / SQL): rows = query,
    columns = partitions, cell = mean time. Helps spot where
    partitioning helps or hurts each query.
    """
    base = mean_std(df, ["query", "partitions", "method"])
    methods = ["dataframe", "sql"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Mean Execution Time Heatmap (seconds)", fontsize=13)

    for ax, method in zip(axes, methods):
        pivot = base[base["method"] == method].pivot(
            index="query", columns="partitions", values="mean"
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_title(method.capitalize())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([q.upper() for q in pivot.index])
        ax.set_xlabel("Partitions")

        for r in range(pivot.values.shape[0]):
            for c in range(pivot.values.shape[1]):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.2f}", ha="center",
                            va="center", fontsize=8,
                            color="black" if val < pivot.values.max() * 0.6 else "white")

        fig.colorbar(im, ax=ax, shrink=0.8, label="seconds")

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "partition_heatmap.png"))


def plot_trial_variance(df, out_dir):
    """
    Box plot of raw trial times per (query, method) to expose outliers
    and whether 3 trials are enough to trust the means.
    """
    queries = sorted(df["query"].unique())
    methods = ["dataframe", "sql"]
    method_colors = {"dataframe": "#2196F3", "sql": "#FF5722"}

    fig, axes = plt.subplots(1, len(queries), figsize=(3 * len(queries), 4), sharey=False)
    fig.suptitle("Trial-to-Trial Variance (all partitions + cache combos)", fontsize=12)

    for ax, q in zip(axes, queries):
        data = [df[(df["query"] == q) & (df["method"] == m)]["time"].values
                for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(method_colors[m])
            patch.set_alpha(0.75)
        ax.set_title(q.upper())
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["DF", "SQL"], fontsize=9)
        ax.set_ylabel("Time (s)" if q == queries[0] else "")
        ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, "trial_variance.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/benchmark.csv",
                        help="CSV produced by benchmarks.py")
    parser.add_argument("--out-dir", default="results",
                        help="Directory to write chart PNGs into")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)

    df["cache"] = df["cache"].astype(bool)
    df["partitions"] = df["partitions"].astype(int)
    df["time"] = df["time"].astype(float)

    print("Generating charts...")
    plot_df_vs_sql_per_query(df, args.out_dir)
    plot_speedup_ratio(df, args.out_dir)
    plot_cache_effect(df, args.out_dir)
    plot_partition_heatmaps(df, args.out_dir)
    plot_trial_variance(df, args.out_dir)
    print("Done.")

if __name__ == "__main__":
    main()