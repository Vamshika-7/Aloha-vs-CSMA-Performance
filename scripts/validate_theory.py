"""Theory validation experiment: Pure ALOHA and Slotted ALOHA vs. Analytical Models.

Evaluates:
- Protocols: Pure ALOHA, Slotted ALOHA
- Node populations: N = 100 (primary), N = 30, N = 10 (comparisons)
- Max retries: max_retries = 10 and max_retries = 50
- Offered load sweep: G in [0.1, 3.0] step 0.1 (30 points)
- Repetitions: 5 independent runs per load point (sim_time = 1000s, base_seed = 42)

Outputs:
- Detailed throughput comparison tables with theoretical S(G)
- Peak throughput identification and interior point verification
- Definition of offered load G in the simulation vs classical theory
"""

import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.metrics import throughput
from simulation.pure_aloha import simulate_pure_aloha
from simulation.slotted_aloha import simulate_slotted_aloha
from simulation.theory import pure_aloha_throughput, slotted_aloha_throughput


def run_experiment_config(
    protocol: str,
    num_nodes: int,
    max_retries: int,
    load_points: List[float],
    num_runs: int = 5,
    sim_time: float = 1000.0,
    packet_duration: float = 1.0,
    propagation_delay: float = 0.01,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Runs a parameter sweep for a specific (protocol, num_nodes, max_retries) tuple."""
    sim_fn = simulate_pure_aloha if protocol == "Pure ALOHA" else simulate_slotted_aloha
    theory_fn = pure_aloha_throughput if protocol == "Pure ALOHA" else slotted_aloha_throughput

    results_per_load = []

    print(f"\n--- Running: {protocol} | N={num_nodes} | max_retries={max_retries} ---")
    t0 = time.time()

    for g in load_points:
        run_tp_list = []
        for r in range(num_runs):
            run_seed = base_seed + r
            res = sim_fn(
                num_nodes=num_nodes,
                sim_time=sim_time,
                offered_load=g,
                packet_duration=packet_duration,
                propagation_delay=propagation_delay,
                max_retries=max_retries,
                seed=run_seed,
            )
            tp = throughput(res["packets"], res["channel_log"], sim_time)
            run_tp_list.append(tp)

        mean_tp = float(np.mean(run_tp_list))
        std_tp = float(np.std(run_tp_list))
        theory_tp = float(theory_fn(g))

        results_per_load.append({
            "load": round(float(g), 2),
            "sim_mean_throughput": round(mean_tp, 4),
            "sim_std_throughput": round(std_tp, 4),
            "theory_throughput": round(theory_tp, 4),
        })

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s")

    # Find peak
    df_res = pd.DataFrame(results_per_load)
    best_idx = df_res["sim_mean_throughput"].idxmax()
    peak_row = df_res.loc[best_idx]
    peak_val = float(peak_row["sim_mean_throughput"])
    peak_g = float(peak_row["load"])

    # Check interior point condition
    is_interior = (load_points[0] < peak_g < load_points[-1])

    return {
        "protocol": protocol,
        "num_nodes": num_nodes,
        "max_retries": max_retries,
        "results": results_per_load,
        "peak_throughput": peak_val,
        "peak_g": peak_g,
        "is_interior": is_interior,
    }


def main():
    print("=" * 70)
    print("INFINITE-POPULATION VALIDATION EXPERIMENT")
    print("Testing Pure ALOHA and Slotted ALOHA against Analytical Theory")
    print("=" * 70)

    load_points = [round(x, 1) for x in np.arange(0.1, 3.05, 0.1)]
    print(f"Offered Loads ({len(load_points)} points): {load_points[0]} to {load_points[-1]}")

    configs = [
        # Primary evaluation: N=100
        ("Pure ALOHA", 100, 10),
        ("Pure ALOHA", 100, 50),
        ("Slotted ALOHA", 100, 10),
        ("Slotted ALOHA", 100, 50),
        # Node scaling comparisons: N=30
        ("Pure ALOHA", 30, 10),
        ("Slotted ALOHA", 30, 10),
        # Node scaling comparisons: N=10
        ("Pure ALOHA", 10, 10),
        ("Slotted ALOHA", 10, 10),
    ]

    all_exp_results = []
    for proto, nodes, retries in configs:
        exp_res = run_experiment_config(
            protocol=proto,
            num_nodes=nodes,
            max_retries=retries,
            load_points=load_points,
            num_runs=5,
            sim_time=1000.0,
            packet_duration=1.0,
            propagation_delay=0.01,
            base_seed=42,
        )
        all_exp_results.append(exp_res)

    # 1. Summary Peak Table
    print("\n" + "=" * 80)
    print("PEAK THROUGHPUT SUMMARY TABLE")
    print("=" * 80)

    summary_rows = []
    for exp in all_exp_results:
        proto = exp["protocol"]
        theo_peak = 0.184 if proto == "Pure ALOHA" else 0.368
        theo_g = 0.5 if proto == "Pure ALOHA" else 1.0

        summary_rows.append({
            "Protocol": proto,
            "Nodes (N)": exp["num_nodes"],
            "Max Retries": exp["max_retries"],
            "Simulated Peak S": f"{exp['peak_throughput']:.4f}",
            "Sim Peak G*": f"{exp['peak_g']:.2f}",
            "Theoretical Peak S": f"{theo_peak:.3f}",
            "Theo Peak G*": f"{theo_g:.1f}",
            "Interior Peak?": "YES" if exp["is_interior"] else "NO (BOUNDARY)",
        })

    df_summary = pd.DataFrame(summary_rows)
    # Print markdown table
    headers = list(df_summary.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, r in df_summary.iterrows():
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    print("\n".join(lines))

    # 2. Detailed Throughput Comparison Table (Selected points)
    print("\n" + "=" * 80)
    print("THROUGHPUT VS LOAD DETAILED COMPARISON TABLE (N=100, max_retries=10)")
    print("=" * 80)

    pure_100 = next(e for e in all_exp_results if e["protocol"] == "Pure ALOHA" and e["num_nodes"] == 100 and e["max_retries"] == 10)
    slotted_100 = next(e for e in all_exp_results if e["protocol"] == "Slotted ALOHA" and e["num_nodes"] == 100 and e["max_retries"] == 10)

    comp_rows = []
    for p_row, s_row in zip(pure_100["results"], slotted_100["results"]):
        g = p_row["load"]
        comp_rows.append({
            "G": f"{g:.1f}",
            "Pure ALOHA (Sim)": f"{p_row['sim_mean_throughput']:.4f}",
            "Pure ALOHA (Theory)": f"{p_row['theory_throughput']:.4f}",
            "Slotted ALOHA (Sim)": f"{s_row['sim_mean_throughput']:.4f}",
            "Slotted ALOHA (Theory)": f"{s_row['theory_throughput']:.4f}",
        })

    df_comp = pd.DataFrame(comp_rows)
    headers = list(df_comp.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, r in df_comp.iterrows():
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    print("\n".join(lines))

    # Save complete raw data to CSV
    csv_rows = []
    for exp in all_exp_results:
        for r in exp["results"]:
            csv_rows.append({
                "protocol": exp["protocol"],
                "num_nodes": exp["num_nodes"],
                "max_retries": exp["max_retries"],
                "offered_load_g": r["load"],
                "sim_mean_throughput": r["sim_mean_throughput"],
                "sim_std_throughput": r["sim_std_throughput"],
                "theory_throughput": r["theory_throughput"],
            })
    out_csv = os.path.join(os.path.dirname(__file__), "theory_validation_results.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\nSaved complete raw results to: {out_csv}")


if __name__ == "__main__":
    main()
