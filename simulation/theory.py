"""Theoretical throughput models and peak comparison for ALOHA protocols.

Pure mathematical formulas for random access channel throughput:
- Pure ALOHA: S = G * exp(-2 * G)
- Slotted ALOHA: S = G * exp(-G)
"""

from typing import Any, Dict, List, Union
import math
import numpy as np
import pandas as pd


def pure_aloha_throughput(G: Union[float, int, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    """Calculates theoretical throughput S for Pure ALOHA given offered load G.

    Formula:
        S = G * exp(-2 * G)

    Theoretical peak occurs at G = 0.5 with S_max = 1 / (2 * e) ≈ 0.1839 (0.184).

    Args:
        G: Offered load (arrivals per packet duration), scalar or array-like.

    Returns:
        Theoretical throughput S. Returns 0.0 for G <= 0.
    """
    if isinstance(G, (int, float)):
        if G <= 0:
            return 0.0
        return float(G * math.exp(-2.0 * G))

    g_arr = np.asarray(G, dtype=float)
    return np.where(g_arr > 0, g_arr * np.exp(-2.0 * g_arr), 0.0)


def slotted_aloha_throughput(G: Union[float, int, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    """Calculates theoretical throughput S for Slotted ALOHA given offered load G.

    Formula:
        S = G * exp(-G)

    Theoretical peak occurs at G = 1.0 with S_max = 1 / e ≈ 0.3679 (0.368).

    Args:
        G: Offered load (arrivals per slot duration), scalar or array-like.

    Returns:
        Theoretical throughput S. Returns 0.0 for G <= 0.
    """
    if isinstance(G, (int, float)):
        if G <= 0:
            return 0.0
        return float(G * math.exp(-G))

    g_arr = np.asarray(G, dtype=float)
    return np.where(g_arr > 0, g_arr * np.exp(-g_arr), 0.0)


def compute_throughput_peaks(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Computes simulated vs theoretical throughput peaks for Pure and Slotted ALOHA.

    Extracts the peak mean throughput and corresponding offered load G from the
    empirical sweep results, and pairs them with theoretical maximum values.

    Args:
        sweep_df: DataFrame containing at least 'protocol', 'load', and 'throughput_mean' columns.

    Returns:
        DataFrame with columns:
            - 'protocol': Protocol name ('Pure ALOHA', 'Slotted ALOHA')
            - 'simulated_peak_throughput': Maximum empirical mean throughput observed
            - 'simulated_peak_g': Offered load G where empirical peak occurred
            - 'theoretical_peak_throughput': Theoretical maximum throughput (0.184 / 0.368)
            - 'theoretical_peak_g': Theoretical optimal offered load G (0.5 / 1.0)
    """
    theoretical_specs = {
        "Pure ALOHA": {
            "theoretical_peak_throughput": round(1.0 / (2.0 * math.e), 3),  # 0.184
            "theoretical_peak_g": 0.5,
        },
        "Slotted ALOHA": {
            "theoretical_peak_throughput": round(1.0 / math.e, 3),  # 0.368
            "theoretical_peak_g": 1.0,
        },
    }

    rows: List[Dict[str, Any]] = []

    for proto in ["Pure ALOHA", "Slotted ALOHA"]:
        spec = theoretical_specs[proto]
        sim_peak = None
        sim_peak_g = None

        if not sweep_df.empty and "protocol" in sweep_df.columns and "throughput_mean" in sweep_df.columns:
            proto_df = sweep_df[sweep_df["protocol"] == proto]
            if not proto_df.empty:
                best_idx = proto_df["throughput_mean"].idxmax()
                best_row = proto_df.loc[best_idx]
                load_col = "load" if "load" in proto_df.columns else ("offered_load_g" if "offered_load_g" in proto_df.columns else "load")
                sim_peak = round(float(best_row["throughput_mean"]), 4)
                sim_peak_g = round(float(best_row[load_col]), 4) if load_col in best_row else None

        rows.append({
            "protocol": proto,
            "simulated_peak_throughput": sim_peak,
            "simulated_peak_g": sim_peak_g,
            "theoretical_peak_throughput": spec["theoretical_peak_throughput"],
            "theoretical_peak_g": spec["theoretical_peak_g"],
        })

    return pd.DataFrame(rows)
