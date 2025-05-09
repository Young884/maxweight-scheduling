#!/usr/bin/env python3
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR

def load_results():
    """
    Load one result (首行) from each CSV in RESULTS_DIR into a DataFrame.
    Parses filename L{links}_F{flows}_{pattern}_{method}.csv for metadata.
    Reads total_load 直接从 CSV 拿，不再重算。
    """
    records = []
    for path in glob.glob(f"{RESULTS_DIR}/*.csv"):
        fname = os.path.basename(path).rstrip(".csv")
        links_s, flows_s, pattern, method = fname.split("_")
        links = int(links_s[1:])
        flows = int(flows_s[1:])

        df = pd.read_csv(path)
        row = df.iloc[0].copy()   # first line selected

        # parse throughputs
        throughputs = eval(row["throughputs"])
        row["SumThroughput"]   = sum(throughputs)
        row["AvgQueue"]        = row["avg_queue"]
        row["AvgDecisionTime"] = row["avg_decision_time"]
        row["Runtime"]         = row["runtime"]

        #total_load
        row["TotalLoad"]       = row["total_load"]

        # metadata
        row["Links"]   = links
        row["Flows"]   = flows
        row["Pattern"] = pattern
        row["Method"]  = method

        records.append(row)

    return pd.DataFrame(records)

def plot_throughput_vs_load_all(df):
    plt.figure(figsize=(8,6))
    for method, grp in df.groupby("Method"):
        grp_sorted = grp.sort_values("TotalLoad")
        #satuation upper bound = min(ceil(Links/2), Flows)
        L = grp_sorted["Links"].iloc[0]
        F = grp_sorted["Flows"].iloc[0]
        sat = min((L + 1)//2, F)

        plt.plot(
            grp_sorted["TotalLoad"],
            grp_sorted["SumThroughput"],
            "o-",
            label=method
        )
        plt.axhline(sat, color="gray", linestyle="--")

    plt.xlabel("Total Arrival Load (∑λᵢ)")
    plt.ylabel("Sum Throughput")
    plt.title("Throughput vs Load by Method (All Configurations)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/throughput_vs_load_all.png")
    plt.close()

def plot_by_LF(df):
    combos = sorted(df[["Links","Flows"]].drop_duplicates().values.tolist())
    cols = 2
    rows = (len(combos) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)

    for idx, (L, F) in enumerate(combos):
        ax = axes[idx//cols][idx%cols]
        sub = df[(df.Links==L)&(df.Flows==F)]
        sat = min((L + 1)//2, F)

        for method, grp in sub.groupby("Method"):
            grp_sorted = grp.sort_values("TotalLoad")
            ax.plot(
                grp_sorted["TotalLoad"],
                grp_sorted["SumThroughput"],
                "o-",
                label=method
            )
        ax.axhline(sat, color="gray", linestyle="--", label=f"Saturation({sat})")

        ax.set_title(f"Links={L}, Flows={F}")
        ax.set_xlabel("Total Arrival Load")
        ax.set_ylabel("Sum Throughput")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(combos), rows*cols):
        fig.delaxes(axes[j//cols][j%cols])

    fig.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/throughput_by_LF.png")
    plt.close()

def plot_queue_vs_method(df):
    plt.figure(figsize=(8,6))
    data = [grp["AvgQueue"].values for _, grp in df.groupby("Method")]
    labels = df["Method"].unique()
    plt.boxplot(data, labels=labels)
    plt.ylabel("Average Queue Length")
    plt.title("Average Queue Length by Method")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/queue_vs_method.png")
    plt.close()

def plot_decision_time(df):
    plt.figure(figsize=(8,6))
    mean_times = df.groupby("Method")["AvgDecisionTime"].mean()
    mean_times.plot.bar()
    plt.ylabel("Avg Decision Time (s)")
    plt.title("Average Scheduling Decision Time by Method")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/decision_time_by_method.png")
    plt.close()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_results()
    # when debuging
    # print(df[["Links","Flows","Pattern","TotalLoad","SumThroughput"]])
    plot_throughput_vs_load_all(df)
    plot_by_LF(df)
    plot_queue_vs_method(df)
    plot_decision_time(df)
    print(f"Plots saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
