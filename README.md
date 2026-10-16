# ALOHA vs CSMA Performance Analyzer

> **Educational Notice**: This software is an interactive, simulation-based Computer Networks performance analyzer and mathematical testbed. It is a **discrete-event mathematical simulation engine**, not a real-world physical packet transmission system.

---

## 1. Problem Statement

In shared-medium broadcast networks (such as satellite links, shared wireless channels, and legacy coaxial Ethernet buses), multiple uncoordinated stations contend for a single communication channel without a central coordinator or scheduler. When two or more stations transmit data packets during overlapping intervals, their electrical or radio signals collide, rendering all overlapping transmissions corrupt and necessitating retransmissions.

Medium Access Control (MAC) protocols define the channel-access rules, carrier-sensing algorithms, and collision-resolution backoff schemes governing how contending stations share the medium. This project models, analyzes, and compares three foundational random-access MAC protocols:
- **Pure ALOHA (Abramson, 1970)**: Uncoordinated, continuous-time transmission where stations transmit immediately upon packet generation without sensing the channel.
- **Slotted ALOHA (Roberts, 1972)**: Discrete time-slotted access where transmissions are restricted to synchronized slot boundaries, halving the vulnerable collision period.
- **1-Persistent CSMA (Carrier Sense Multiple Access, Kleinrock & Tobagi, 1975)**: "Listen-before-talk" carrier sensing where nodes defer transmission while the channel is busy, yet remain subject to collisions caused by finite signal propagation delay and post-busy transmission accumulation.

---

## 2. Objectives

1. **Faithful Physical & Temporal Mechanics**:
   - Model continuous-time overlap detection for Pure ALOHA.
   - Enforce exact slot-boundary synchronization for Slotted ALOHA.
   - Implement carrier sensing with propagation delay vulnerability windows ($\tau$) and post-busy clear accumulation collisions for 1-Persistent CSMA.
2. **Deterministic Pseudorandomness & Independent Streams**:
   - Eliminate global random state by isolating Poisson traffic generation and per-protocol exponential backoff into dedicated NumPy `SeedSequence` streams (`randomness.py`).
3. **Controlled Multi-Protocol Comparison**:
   - Evaluate all three protocols against an **identical generated packet arrival trace** under identical seeds and parameters.
4. **Automated Invariant Verification**:
   - Enforce a 10-check consistency suite (`consistency.py`) covering packet accounting, log alignment, channel physics, metric bounds, causality, reproducibility, and cross-path equivalence.
5. **Analytical Theory Benchmarking**:
   - Overlay theoretical throughput curves ($S = G \cdot e^{-2G}$ and $S = G \cdot e^{-G}$) and compare empirical peaks with analytical limits.
6. **Self-Describing Data Portability**:
   - Provide comprehensive UTF-8 CSV exports with complete parameter provenance across all application modes (`export.py`).

---

## 3. Architecture & Project Structure

```
aloha-csma-analyzer/
├── requirements.txt            # Pinned package dependencies
├── README.md                   # System documentation and theoretical reference
├── REPORT_NOTES.md             # Empirical findings from fresh baseline and sweep runs
├── app.py                      # Streamlit interactive multi-tab analytical dashboard
├── scripts/                    # Validation and experiment runner scripts
│   ├── validate_theory.py      # Infinite-population theory validation experiment runner
│   └── theory_validation_results.csv # Empirical sweep dataset across N in {10, 30, 100}
├── simulation/                 # Discrete-event simulation and analytical package
│   ├── __init__.py             # Package exports for data structures, protocols, and metrics
│   ├── base.py                 # Packet, Node, Channel data classes and status enums
│   ├── traffic.py              # Synthetic Poisson packet generator based on offered load G
│   ├── pure_aloha.py           # Continuous-time Pure ALOHA discrete-event engine
│   ├── slotted_aloha.py        # Discrete-slot ALOHA simulation engine
│   ├── csma.py                 # 1-Persistent CSMA with carrier sensing and propagation delay
│   ├── metrics.py              # Pure metric calculation functions with documented denominators
│   ├── consistency.py          # 10-check invariant verification, physical validation, and cross-path equivalence
│   ├── theory.py               # Analytical throughput functions and peak comparison helpers
│   ├── export.py               # Self-describing CSV export DataFrame builders
│   ├── randomness.py           # Deterministic NumPy SeedSequence stream manager
│   └── sweep.py                # Identical-trace comparison and multi-run parametric sweeps
├── visualization/              # Data visualization package
│   ├── __init__.py             # Visualization package exports
│   ├── timeline.py             # Plotly horizontal per-node channel activity Gantt timeline
│   └── sweep_plots.py          # Plotly parametric sweep plots with clipped error bars (μ ± σ)
└── tests/                      # Automated pytest suite (78 passing unit & integration tests)
    ├── __init__.py             # Test discovery marker
    ├── test_aloha.py           # Pure ALOHA and Slotted ALOHA collision & boundary tests (9 tests)
    ├── test_csma.py            # CSMA carrier sensing and propagation delay tests (5 tests)
    ├── test_metrics.py         # Hand-computed reference validation for all metric functions (15 tests)
    ├── test_consistency.py     # Invariant checker and intentional failure detection tests (17 tests)
    ├── test_sweep.py           # Comparison mode trace sharing, sweeps, and statistics tests (9 tests)
    ├── test_theory.py          # Analytical formula limits and peak-finding tests (4 tests)
    ├── test_export.py          # Self-describing CSV export DataFrame integrity tests (6 tests)
    └── test_ui.py              # Headless Streamlit AppTest integration and download tests (13 tests)
```

---

## 4. Randomness and Reproducibility

### Independent Seeded Streams (`simulation/randomness.py`)

A fundamental requirement for rigorous scientific benchmarking is that random number generation must be deterministic, isolated, and invariant across execution modes. Global random seeds or uncoordinated RNG draws cause coupling artifacts where changing one parameter or evaluating an additional protocol shifts random draws for subsequent protocols.

To prevent this, [`simulation/randomness.py`](file:///c:/Users/mitta/Aloha-vs-CSMA-Performance/simulation/randomness.py) partitions pseudorandom number generation using NumPy's modern `SeedSequence` spawning architecture:

```python
_STREAMS = {
    "traffic": 0,
    "Pure ALOHA": 1,
    "Slotted ALOHA": 2,
    "1-Persistent CSMA": 3,
}

def stream_rng(seed: Seed, stream: str, supplied_rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    if supplied_rng is not None:
        return supplied_rng
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(_STREAMS[stream],))
    )
```

### Why Single Simulation and Comparison Mode Produce Identical Results

1. **Traffic Generation Isolation (`stream="traffic"`, `spawn_key=(0,)`)**:
   In both Single Simulation and Comparison mode, Poisson arrival times and source node assignments are drawn exclusively from the traffic RNG stream (`spawn_key=(0,)`). The generated arrival sequence depends solely on the base seed, node count, duration, and offered load.
2. **Backoff Generation Isolation (`stream=protocol`, `spawn_key=(1, 2, or 3)`)**:
   When collisions occur, retransmission backoff intervals are drawn exclusively from that protocol's private backoff RNG stream (`spawn_key=(1,)` for Pure ALOHA, `(2,)` for Slotted ALOHA, and `(3,)` for 1-Persistent CSMA).
3. **Absence of Cross-Stream Interference**:
   Because traffic generation and protocol backoffs consume distinct spawned streams, evaluating Slotted ALOHA or CSMA in Comparison mode never consumes numbers from Pure ALOHA's backoff stream. Consequently:
   $$\text{Single Simulation (Pure ALOHA, Seed } S) \equiv \text{Comparison Mode (Pure ALOHA, Seed } S)$$
   Both execution paths produce bitwise identical packet generation times, attempt times, successful times, retry counts, and channel event logs.

---

## 5. Performance Metrics & Exact Formulas

All metrics are implemented as pure mathematical functions in [`simulation/metrics.py`](file:///c:/Users/mitta/Aloha-vs-CSMA-Performance/simulation/metrics.py). The exact formulas, numerators, and denominators defined in the codebase are:

### 1. Throughput
- **Definition**: Successfully delivered packets per second of simulation time.
- **Formula**:
  $$\text{Throughput} = \frac{\text{Number of successfully delivered packets}}{\text{sim\_time}}$$
- **Denominator**: `sim_time` (total simulation duration in seconds). Returns `0.0` if `sim_time <= 0`.

### 2. Normalized Throughput ($S$)
- **Definition**: Dimensionless channel efficiency representing the fraction of channel capacity carrying successful data.
- **Formula**:
  $$\text{Normalized Throughput} = \frac{\text{successful\_packets} \times \text{packet\_duration}}{\text{sim\_time}} = \text{Throughput} \times \text{packet\_duration}$$
- **Denominator**: `sim_time / packet_duration` (total transmission slot capacity over the simulation window). Returns `0.0` if `sim_time <= 0` or `packet_duration <= 0`.

### 3. Collision Rate
- **Definition**: The empirical probability that a transmission attempt on the channel results in a collision.
- **Formula**:
  $$\text{Collision Rate} = \frac{\text{Number of COLLISION events in channel\_log}}{\text{Total transmission attempts in channel\_log}}$$
- **Numerator**: Exact count of `COLLISION` entries in `channel_log`.
- **Denominator**: `len(channel_log)` (total transmission attempts logged on the channel). Returns `0.0` if `len(channel_log) == 0`.

### 4. Success Rate
- **Definition**: The end-to-end packet delivery ratio.
- **Formula**:
  $$\text{Success Rate} = \frac{\text{Number of successfully delivered packets}}{\text{Total generated packets}}$$
- **Denominator**: `len(packets)` (total unique packets generated across all nodes). Returns `0.0` if `len(packets) == 0`.

### 5. Average Delay
- **Definition**: The mean latency elapsed from packet generation to successful delivery completion.
- **Formula**:
  $$\text{Average Delay} = \frac{\sum_{p \in \text{SUCCESS}} (t_{\text{successful}} - t_{\text{generation}})}{\text{Number of successfully delivered packets}}$$
- **Denominator**: Number of successfully delivered packets (`len([p for p in packets if p.status == SUCCESS])`). Returns `0.0` if no packets were successfully delivered.

### 6. Average Retransmissions
- **Definition**: The mean number of collision retransmissions incurred per generated packet.
- **Formula**:
  $$\text{Average Retransmissions} = \frac{\sum_{p \in \text{packets}} p.\text{retry\_count}}{\text{Total generated packets}}$$
- **Denominator**: `len(packets)` (total unique packets generated). Returns `0.0` if `len(packets) == 0`.

### 7. Dropped Packets
- **Definition**: The total count of packets permanently discarded due to exceeding the retry threshold.
- **Formula**:
  $$\text{Dropped Packets} = \sum_{p \in \text{packets}} \mathbb{I}(p.\text{status} == \text{DROPPED})$$

### 8. Channel Utilization (Useful Airtime Only)
- **Definition**: The fraction of simulation time the channel was occupied by **successful** transmissions. Collision intervals are strictly excluded so that utilization measures only useful data transfer time. Overlapping intervals (if any) are merged.
- **Formula**:
  $$\text{Channel Utilization} = \frac{\text{Total duration of merged SUCCESS transmission intervals}}{\text{sim\_time}}$$
- **Denominator**: `sim_time`. Returns `0.0` if `sim_time <= 0` or if there are no successful transmissions.

### 9. Channel Busy Fraction (All Airtime Including Collisions)
- **Definition**: The total fraction of simulation time the channel carried transmission energy across all attempts (both `SUCCESS` and `COLLISION`). Overlapping transmission intervals are merged to avoid double-counting channel occupancy.
- **Formula**:
  $$\text{Channel Busy Fraction} = \frac{\text{Total busy duration of all merged transmission intervals}}{\text{sim\_time}}$$
- **Denominator**: `sim_time`. Returns `0.0` if `sim_time <= 0` or if `channel_log` is empty. Always satisfies $\text{Channel Busy Fraction} \ge \text{Channel Utilization}$.

---

## 6. Validation & Invariant Checking

### The 10 Consistency Checks (`simulation/consistency.py`)

To ensure simulation integrity and detect subtle state-tracking regressions, the engine executes [`run_consistency_checks`](file:///c:/Users/mitta/Aloha-vs-CSMA-Performance/simulation/consistency.py#L313) across every run:

1. **Packet Accounting**: Verifies that total packets generated equals the sum of successful, dropped, and pending packets:
   $$\text{packets\_generated} == \text{successful} + \text{dropped} + \text{pending}$$
2. **Per-Node vs Packet-Level Counts**: Confirms that the sum of per-node generated and successful counts equals the global packet-level counts:
   $$\sum_{n} n.\text{packets\_generated} == \text{len(packets)}, \quad \sum_{n} n.\text{packets\_successful} == \text{successful\_count}$$
3. **Channel Log SUCCESS Alignment**: Verifies that the count of `SUCCESS` events in `channel_log` equals the number of packets with `status == SUCCESS`.
4. **Collision Count and Metric Consistency**: Confirms that `COLLISION` events in `channel_log` equal `collision_attempts` and exactly match the fraction used to compute `collision_rate`.
5. **Total Retransmissions vs Packet Retries**: Asserts that total node retransmissions equal the sum of individual packet retry counters:
   $$\sum_{n} n.\text{retransmissions} == \sum_{p} p.\text{retry\_count}$$
6. **Metric Range and Validity Bounds**: Validates that all rates and fractions (`collision_rate`, `success_rate`, `channel_utilization`, `channel_busy_fraction`) reside in $[0, 1]$, non-negative metrics (`throughput`, `avg_delay`, `avg_retransmissions`) are $\ge 0$, and enforces the physical invariant:
   $$\text{channel\_busy\_fraction} \ge \text{channel\_utilization}$$
7. **Temporal Causality**: Asserts that every delivered packet completed transmission after its creation timestamp:
   $$t_{\text{successful}} \ge t_{\text{generation}} \quad \forall p \in \text{SUCCESS}$$
8. **Deterministic Reproducibility (Seed Invariance)**: Re-runs the simulation with an identical seed and configuration, verifying bitwise identical packet records and channel logs.
9. **Cross-Path Equivalence**: Asserts that executing a protocol via the Single Simulation path yields identical packets, channel logs, and performance metrics as executing it within Comparison mode under the same parameters.
10. **Channel Log Physical Invariants**: Performs a strict, protocol-independent physics audit directly on the event intervals in `channel_log`:
    - *Pure ALOHA*: Asserts that no `SUCCESS` transmission interval overlaps in time with any other transmission interval, and any two overlapping intervals are both marked `COLLISION`.
    - *Slotted ALOHA*: Asserts that every transmission starts precisely on a discrete slot boundary ($t \pmod T < 10^{-6}$), no slot carries multiple transmissions unless all are marked `COLLISION`, and any slot with $\ge 2$ entries has all of them marked `COLLISION`.
    - *1-Persistent CSMA*: Asserts that no `SUCCESS` transmission overlaps with another transmission whose start time exceeds the channel vulnerable period ($\tau = \text{propagation\_delay}$). Any transmission starting after $\tau$ into an existing transmission violates carrier sensing.

---

### Theoretical Throughput Overlay (`simulation/theory.py`)

In the **Load Analysis** tab, the Offered Load vs Throughput chart provides a toggleable theoretical overlay:
- **Pure ALOHA (theoretical)**:
  $$S = G \cdot e^{-2G}, \quad S_{\max} = \frac{1}{2e} \approx 0.1839 \text{ at } G = 0.5$$
- **Slotted ALOHA (theoretical)**:
  $$S = G \cdot e^{-G}, \quad S_{\max} = \frac{1}{e} \approx 0.3679 \text{ at } G = 1.0$$
- **1-Persistent CSMA**: No theoretical curve is drawn. The Abramson/Roberts ALOHA exponential models do not apply to carrier-sensing protocols, whose throughput depends strongly on the normalized propagation delay parameter $a = \tau / T$ (Kleinrock-Tobagi formulation).

Curves are rendered as 200-point smooth dashed lines in the matching protocol color families (`#3B82F6` for Pure ALOHA, `#F59E0B` for Slotted ALOHA).

#### Why Simulated Curves Differ from Analytical Theory & Large-Population Validation

Under the throughput plot, the application explicitly displays this explanatory note:
> *Simulated values can differ from theory because the simulation has a finite number of nodes, per-node queues, a retry limit and finite duration, while the formulas assume an infinite population and Poisson offered traffic that includes retransmissions.*

1. **Finite Population Smoothing ($N = 10, 30$)**:
   Classical theory assumes an infinite population ($N \to \infty$) of uncoordinated stations. In a finite network with per-node queues, a backlogged node holding a packet cannot generate new packets. Furthermore, colliding nodes enter binary exponential backoff ($U(0, 2^{\min(k, 10)} \cdot T)$), spreading retransmissions over time and providing implicit congestion control that prevents destructive collision cascades at moderate loads.
2. **Infinite-Population Validation ($N = 100$)**:
   When simulated under asymptotic conditions approximating an unconstrained population ($N = 100$, $T_{\text{sim}} = 1000\,\text{s}$, $T = 1.0\,\text{s}$, $\tau = 0.01\,\text{s}$ across $G \in [0.1, 3.0]$ via `scripts/validate_theory.py`):
   - **Pure ALOHA** achieves an interior peak mean throughput of **$S = 0.1660$** ($k_{\max}=10$) and **$S = 0.1710$** ($k_{\max}=50$) at $G^* = 0.20$, reaching **$90.3\% - 93.0\%$** of the theoretical maximum ($1/(2e) \approx 0.1839$).
   - **Slotted ALOHA** achieves an interior peak mean throughput of **$S = 0.3322$** ($k_{\max}=10$) and **$S = 0.3270$** ($k_{\max}=50$) at $G^* = 0.40$, reaching **$90.3\%$** of the theoretical maximum ($1/e \approx 0.3679$).
   - Both protocols exhibit genuine interior peaks followed by steep throughput collapse toward zero as load intensifies, matching the signature shape of classical ALOHA instability.
3. **Offered Load Definition ($G_{\text{new}}$ vs $G_{\text{total}}$)**:
   - In the simulation, $G = \lambda_{\text{new}} \cdot T$ represents newly generated Poisson arrivals.
   - In Abramson/Roberts analytical theory, $G = (\lambda_{\text{new}} + \lambda_{\text{retry}}) \cdot T$ represents total aggregate channel attempts including all retransmissions.
   Because collision retransmissions compound channel contention, the channel experiences critical saturation ($G_{\text{total}} \approx 0.5$ and $1.0$) when newly offered load reaches $G_{\text{new}} \approx 0.20$ and $0.40$.

---

## 7. Self-Describing CSV Exports (`simulation/export.py`)

Every export generated by the application includes full parameter metadata (`num_nodes`, `sim_time_s`, `offered_load_g`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed`, `protocol`) and standardized column headers with explicit units:

| Tab | Export Button Label | Generator Function | Exported Columns |
| :--- | :--- | :--- | :--- |
| **Tab 1: Dashboard** | `📥 Download Metrics Summary (CSV)` | `export_metrics_dataframe` | `protocol`, `num_nodes`, `sim_time_s`, `offered_load_g`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed`, `throughput_pkts_per_s`, `normalized_throughput_s`, `collision_rate`, `collision_attempts`, `collision_affected_packets`, `success_rate`, `avg_delay_s`, `avg_retransmissions`, `dropped_packets`, `pending_packets`, `channel_utilization`, `channel_busy_fraction` |
| **Tab 1: Dashboard** | `📥 Download Consistency Results (CSV)` | `export_consistency_dataframe` | `protocol`, `num_nodes`, `sim_time_s`, `offered_load_g`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed`, `check_name`, `passed`, `status`, `details` |
| **Tab 2: Simulation** | `📥 Download Packet-Level Table (CSV)` | `export_packets_dataframe` | `protocol`, `num_nodes`, `sim_time_s`, `offered_load_g`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed`, `packet_id`, `source_node`, `generation_time_s`, `attempt_time_s`, `successful_time_s`, `retry_count`, `status` |
| **Tab 4: Comparison** | `📥 Download Comparison Metrics (CSV)` | `export_comparison_dataframe` | `metric`, `pure_aloha`, `slotted_aloha`, `csma_1_persistent`, `num_nodes`, `sim_time_s`, `offered_load_g`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed` |
| **Tab 4: Comparison** | `📥 Download Comparison Consistency Results (CSV)` | `export_consistency_dataframe` | `protocol`, `num_nodes`, `sim_time_s`, `offered_load_g`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed`, `check_name`, `passed`, `status`, `details` |
| **Tab 5: Load Analysis** | `📥 Download Raw Load Sweep Results (CSV)` | `export_sweep_raw_dataframe` | `protocol`, `offered_load_g`, `num_nodes`, `seed`, `run_index`, `sim_time_s`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `throughput_pkts_per_s`, `normalized_throughput_s`, `collision_rate`, `collision_attempts`, `collision_affected_packets`, `success_rate`, `avg_delay_s`, `avg_retransmissions`, `dropped_packets`, `pending_packets`, `channel_utilization`, `channel_busy_fraction` |
| **Tab 5: Load Analysis** | `📥 Download Aggregated Load Sweep Summary (CSV)` | `export_sweep_aggregated_dataframe` | `load`, `protocol`, `num_runs`, `throughput_mean`, `throughput_std`, `normalized_throughput_mean`, `normalized_throughput_std`, `collision_rate_mean`, `collision_rate_std`, `success_rate_mean`, `success_rate_std`, `avg_delay_mean`, `avg_delay_std`, `avg_retransmissions_mean`, `avg_retransmissions_std`, `dropped_packets_mean`, `dropped_packets_std`, `channel_utilization_mean`, `channel_utilization_std`, `channel_busy_fraction_mean`, `channel_busy_fraction_std`, plus metadata parameters `num_nodes`, `sim_time_s`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed` |
| **Tab 5: Load Analysis** | `📥 Download Raw Node Sweep Results (CSV)` | `export_sweep_raw_dataframe` | `protocol`, `num_nodes`, `offered_load_g`, `seed`, `run_index`, `sim_time_s`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `throughput_pkts_per_s`, `normalized_throughput_s`, `collision_rate`, `collision_attempts`, `collision_affected_packets`, `success_rate`, `avg_delay_s`, `avg_retransmissions`, `dropped_packets`, `pending_packets`, `channel_utilization`, `channel_busy_fraction` |
| **Tab 5: Load Analysis** | `📥 Download Aggregated Node Sweep Summary (CSV)` | `export_sweep_aggregated_dataframe` | `num_nodes`, `protocol`, `num_runs`, `throughput_mean`, `throughput_std`, `normalized_throughput_mean`, `normalized_throughput_std`, `collision_rate_mean`, `collision_rate_std`, `success_rate_mean`, `success_rate_std`, `avg_delay_mean`, `avg_delay_std`, `avg_retransmissions_mean`, `avg_retransmissions_std`, `dropped_packets_mean`, `dropped_packets_std`, `channel_utilization_mean`, `channel_utilization_std`, `channel_busy_fraction_mean`, `channel_busy_fraction_std`, plus metadata parameters `offered_load_g`, `sim_time_s`, `packet_duration_s`, `propagation_delay_s`, `max_retries`, `seed` |

---

## 8. User Interface & Screenshot Placeholders

The application interface is structured across seven dedicated tabs:

1. **Dashboard**
   - *Placeholder: `docs/screenshots/dashboard.png`*
   - Displays real-time KPI metric tiles, throughput gauges, channel utilization vs. busy fraction cards, and the summary CSV export button.
2. **Consistency Checks Expander**
   - *Placeholder: `docs/screenshots/consistency_checks.png`*
   - Expandable audit panel rendering real-time pass/fail status and detailed diagnostic messages for all 10 invariant checks.
3. **Simulation Trace Table**
   - *Placeholder: `docs/screenshots/simulation_trace.png`*
   - Interactive, searchable packet lifecycle inspection table with status filtering (`SUCCESS`, `DROPPED`, `PENDING`) and packet CSV export.
4. **Channel Activity Timeline**
   - *Placeholder: `docs/screenshots/timeline.png`*
   - Plotly horizontal Gantt timeline illustrating per-node transmission activity, contention intervals, and collisions over time.
5. **Protocol Comparison Table**
   - *Placeholder: `docs/screenshots/comparison_table.png`*
   - Side-by-side performance matrix evaluating Pure ALOHA, Slotted ALOHA, and 1-Persistent CSMA against an identical traffic trace.
6. **Load Analysis (Theory Overlay & Peak Table)**
   - *Placeholder: `docs/screenshots/load_analysis_theory.png`*
   - Parametric sweep line plots with error bars ($\mu \pm \sigma$), toggleable analytical overlay curves, and empirical vs. theoretical peak throughput comparison table.
7. **Node Count Sweep**
   - *Placeholder: `docs/screenshots/node_sweep.png`*
   - Contending station scalability sweep evaluating throughput sensitivity as node density increases at a fixed offered load.

---

## 9. Installation & Test Execution

### Prerequisites
- Python 3.10+ (tested on Python 3.12 and 3.14)
- Git

### Setup
```powershell
git clone https://github.com/your-repo/aloha-csma-analyzer.git
cd aloha-csma-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Launching the Web Application
```powershell
python -m streamlit run app.py
```
Access the application at `http://localhost:8501`.

### Running Automated Test Suite
```powershell
python -m pytest -v
```

**Current Test Status**: **78 passed in ~17 seconds** across 8 test suites:
- `tests/test_aloha.py`: 9 passed
- `tests/test_csma.py`: 5 passed
- `tests/test_metrics.py`: 15 passed
- `tests/test_consistency.py`: 17 passed
- `tests/test_sweep.py`: 9 passed
- `tests/test_theory.py`: 4 passed
- `tests/test_export.py`: 6 passed
- `tests/test_ui.py`: 13 passed
