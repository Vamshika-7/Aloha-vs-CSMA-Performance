# ALOHA vs CSMA Performance Analyzer

> An interactive discrete-event simulation engine and analytical testbed comparing random-access MAC protocols: Pure ALOHA, Slotted ALOHA, and 1-Persistent CSMA.

---

## 1. Problem Statement

In shared-medium computer and wireless networks (e.g., satellite links, legacy Ethernet buses, shared wireless channels), multiple independent nodes transmit data frames across a single broadcast channel without a centralized scheduler. When two or more nodes transmit frames in overlapping time intervals, their electrical or radio signals collide, corrupting all overlapping transmissions and requiring retransmissions.

Medium Access Control (MAC) protocols govern how contending stations decide when to transmit, sense the channel, and resolve collisions. This project empirically analyzes the evolution and performance trade-offs of three foundational random-access MAC protocols:
- **Pure ALOHA**: Uncoordinated immediate transmission with zero channel sensing.
- **Slotted ALOHA**: Time-slotted synchronization that halves the vulnerable collision window.
- **1-Persistent CSMA (Carrier Sense Multiple Access)**: "Listen-before-talk" carrier sensing that defers transmission when the channel is busy, while remaining subject to collisions caused by finite signal propagation delay.

---

## 2. Objectives

1. **Faithful Collision Modeling**: Implement exact physical and temporal collision rules for Pure ALOHA (continuous temporal overlap), Slotted ALOHA (discrete slot collisions), and 1-Persistent CSMA (propagation delay vulnerability window and post-busy accumulation collisions).
2. **Deterministic & Reproducible Experiments**: Ensure all pseudorandom traffic arrivals and exponential backoff delays flow through explicit seeded NumPy generators (`np.random.default_rng(seed)`), avoiding global random state.
3. **Controlled Multi-Protocol Comparison**: Provide a comparison mode that feeds an **identical packet arrival trace** to all three protocols under the exact same load and seeds.
4. **Parametric Scaling & Statistical Rigor**: Support parameter sweeps across offered loads ($G$) and node counts ($N$) with multi-run statistical confidence bounds (mean $\pm$ standard deviation).
5. **Interactive Exploration**: Deliver a responsive Streamlit web interface with interactive Gantt-style channel activity timelines and packet-level inspection.

---

## 3. Technologies Used

- **Python 3.10+**: Core programming language.
- **NumPy (`2.5.3`)**: Seeded pseudorandom number generation (Poisson arrivals, exponential backoff) and vector statistics.
- **Pandas (`3.0.5`)**: Tabular packet logging, aggregation, and data structuring.
- **Plotly (`7.1.0`)**: Interactive Gantt-style channel transmission timelines and parametric sweep line charts with error bars.
- **Matplotlib (`3.11.2`)**: Backend plotting support.
- **Streamlit (`1.64.0`)**: Web-based analytical dashboard, interactive widgets, `@st.cache_data` caching, and session state management.
- **Pytest (`9.1.1`)**: Test suite covering protocol mechanics, metric accuracy, identical-trace comparisons, and UI integration.

---

## 4. Architecture & Folder Structure

```
aloha-csma-analyzer/
├── requirements.txt            # Exact pinned package dependencies
├── README.md                   # Comprehensive architectural and usage documentation
├── app.py                      # Streamlit interactive web application and state orchestrator
├── simulation/                 # Core discrete-event simulation package
│   ├── __init__.py             # Exports data structures, protocol runners, metrics, and sweep helpers
│   ├── base.py                 # Packet, Node, and Channel data classes and status enums
│   ├── traffic.py              # Poisson synthetic packet generation based on offered load G
│   ├── pure_aloha.py           # Continuous-time Pure ALOHA discrete-event simulation
│   ├── slotted_aloha.py        # Discrete-slot ALOHA simulation
│   ├── csma.py                 # 1-Persistent CSMA with carrier sensing and propagation delay
│   ├── metrics.py              # Pure metric calculation functions with documented denominators
│   └── sweep.py                # Identical-trace comparison, parameter sweeps, and multi-run statistics
├── visualization/              # Data visualization package
│   ├── __init__.py             # Exports timeline and sweep plotting routines
│   ├── timeline.py             # Plotly horizontal per-node channel activity Gantt timeline
│   └── sweep_plots.py          # Plotly multi-line sweep figures with error bars (mean ± std)
└── tests/                      # Automated test suite (32 unit & integration tests)
    ├── __init__.py             # Package marker for test discovery
    ├── test_aloha.py           # Pure ALOHA and Slotted ALOHA collision and boundary tests
    ├── test_csma.py            # CSMA carrier sensing and propagation delay tuning tests
    ├── test_metrics.py         # Hand-computed reference verification for all 8 metrics
    ├── test_sweep.py           # Verification of identical traces, sweep structure, and mean/std math
    └── test_ui.py              # Streamlit AppTest end-to-end integration tests
```

---

## 5. Protocol Descriptions & Collision Rules

### Pure ALOHA
- **Working Principle**: Fully uncoordinated and decentralized continuous-time transmission. Whenever a node has a packet ready in its queue, it transmits immediately without sensing whether the channel is currently occupied.
- **Collision Rule**: Any temporal overlap between two or more transmissions on the shared channel destroys all overlapping packets. For a packet transmitted at time $s$ of duration $T$, any other transmission starting in $(s - T, s + T)$ results in a collision.
- **Backoff & Drop**: Colliding nodes increment their retry count $k$ and pick a uniform continuous random backoff $t_{\text{backoff}} \sim U(0, 2^{\min(k, 10)} \cdot T)$. If $k > \text{max\_retries}$, the packet is dropped.

### Slotted ALOHA
- **Working Principle**: Channel time is partitioned into discrete, synchronized slots of length $T = \text{packet\_duration}$. Nodes may only initiate transmissions at exact slot boundaries ($m \cdot T$). A packet arriving mid-slot must wait in queue until the next slot boundary $\lceil t_{\text{gen}} / T \rceil$.
- **Collision Rule**: A collision occurs if and only if two or more nodes attempt transmission within the *same* slot. A single transmission within a slot is guaranteed to succeed.
- **Backoff & Drop**: Colliding nodes pick an integer future slot $m + R$ where $R \in [1, 2^{\min(k, 10)}]$. If $k > \text{max\_retries}$, the packet is dropped.

### 1-Persistent CSMA (Carrier Sense Multiple Access)
- **Working Principle**: "Listen before talk". A node with a packet senses the shared channel:
  - If the channel is sensed **IDLE**: Transmit immediately.
  - If the channel is sensed **BUSY**: Continuously listen until the channel becomes idle, then transmit immediately with probability $p = 1$.
- **Collision Rule**:
  1. *Propagation Delay Vulnerability Window ($\tau$)*: Signals propagate at finite velocity. If Node A begins transmitting at $s_A$, its signal reaches Node B at $s_A + \tau$. If Node B senses the channel at $s_B \in [s_A, s_A + \tau)$, Node B perceives the channel as IDLE and transmits, causing a collision: $|s_A - s_B| < \tau$.
  2. *Accumulation During Busy Channel*: If two or more nodes have packets arrive while another node is transmitting, all of them wait. The instant the active transmission clears, all waiting nodes detect an idle channel simultaneously, transmit at the same instant ($|s_1 - s_2| = 0$), and collide.
- **Backoff & Drop**: Colliding nodes back off by $t_{\text{backoff}} \sim U(0, 2^{\min(k, 10)} \cdot T)$ and then re-sense the channel.

---

## 6. Simulation Methodology

### Discrete-Event Engine
Simulations are driven by an efficient priority queue (`heapq`) handling timestamped events:
- `PACKET_ARRIVAL`: Enqueues newly generated packets.
- `SENSE_AND_TRANSMIT`: Performs carrier sensing in CSMA.
- `CHANNEL_CLEAR_CHECK`: Wakes waiting nodes when active transmission signals clear.
- `TRANSMIT_END`: Evaluates collision status, updates metrics, and schedules retries.

### Seeded Pseudorandomness
No global random state or bare `random` module calls are used. Every stochastic operation (exponential inter-arrival times and backoff windows) flows through an explicitly passed `np.random.default_rng(seed)` instance.

### Identical Traffic Trace in Comparison Mode
To ensure completely fair and objective comparisons:
1. One shared packet arrival trace is generated from `traffic.py` using the selected seed.
2. The trace contains fixed arrival timestamps and assigned source nodes.
3. The trace is deeply cloned into clean, unmutated instances (`clone_trace()`) and fed identically into `simulate_pure_aloha()`, `simulate_slotted_aloha()`, and `simulate_csma()`.
4. Differences in throughput, delay, and collision rates emerge purely from protocol logic, not from random traffic variation.

---

## 7. Performance Metrics & Exact Formulas

All metrics are implemented as pure functions in [`simulation/metrics.py`](file:///c:/Users/vamsh/Downloads/CN/aloha-csma-analyzer/simulation/metrics.py) with explicit denominators documented:

| Metric | Formula | Denominator Used | Purpose |
| :--- | :--- | :--- | :--- |
| **Throughput** | $\frac{\text{Successful Packets}}{\text{sim\_time}}$ | $\text{sim\_time}$ (total simulation seconds) | Successfully delivered packets per unit time |
| **Normalized Throughput ($S$)** | $\frac{\text{Successful Packets} \times T}{\text{sim\_time}} = \text{Throughput} \times T$ | $\frac{\text{sim\_time}}{T}$ (total transmission slot capacity) | Dimensionless channel efficiency ($0 \le S \le 1$) |
| **Collision Rate** | $\frac{\text{Number of COLLISION entries in channel\_log}}{\text{Total entries in channel\_log}}$ | `len(channel_log)` (total transmission attempts) | Probability that a channel attempt suffers collision |
| **Success Rate** | $\frac{\text{Successful Packets}}{\text{Total Generated Packets}}$ | `len(packets)` (total unique packets generated) | End-to-end packet delivery ratio |
| **Average Delay** | $\frac{\sum (t_{\text{successful}} - t_{\text{generation}})}{\text{Successful Packets}}$ | Count of successful packets | Mean end-to-end latency for delivered packets |
| **Average Retransmissions**| $\frac{\sum \text{packet.retry\_count}}{\text{Total Generated Packets}}$ | `len(packets)` (total unique packets generated) | Mean retransmission burden per generated packet |
| **Dropped Packets** | Count of packets with `status == DROPPED` | Count (integer) | Total packets discarded after exceeding `max_retries` |
| **Channel Utilization** | $\frac{\text{Total Busy Duration of Merged Intervals}}{\text{sim\_time}}$ | $\text{sim\_time}$ (total simulation seconds) | Fraction of time channel carried transmission signals |

---

## 8. Mathematical Theory vs. Empirical Simulation

> **Important Distinction**: The formulas below represent analytical mathematical models under infinite-node Poisson assumptions. They serve as theoretical reference baselines and are distinct from empirical discrete-event simulation outputs.

### Pure ALOHA Theoretical Model
- **Vulnerable Period**: $V_{\text{pure}} = 2T$
- **Throughput Equation**:
  $$S = G \cdot e^{-2G}$$
- **Maximum Theoretical Throughput**: Differentiating with respect to $G$ yields maximum at $G = 0.5$:
  $$S_{\max} = \frac{1}{2e} \approx 0.1839 \quad (18.4\%)$$

### Slotted ALOHA Theoretical Model
- **Vulnerable Period**: $V_{\text{slotted}} = T$
- **Throughput Equation**:
  $$S = G \cdot e^{-G}$$
- **Maximum Theoretical Throughput**: Differentiating with respect to $G$ yields maximum at $G = 1.0$:
  $$S_{\max} = \frac{1}{e} \approx 0.3679 \quad (36.8\%)$$

### 1-Persistent CSMA Theoretical Model (Kleinrock & Tobagi)
- **Vulnerable Period**: $V_{\text{csma}} = \tau$ (normalized delay parameter $a = \tau / T$)
- **Behavior**: As $a \to 0$, vulnerable period approaches 0 and theoretical throughput exceeds $50\%$ to $80\%$ under moderate loads, dramatically outperforming both ALOHA variants.

---

## 9. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git (optional)

### Setup Instructions
1. Navigate to the project root directory:
   ```powershell
   cd c:\Users\vamsh\Downloads\CN\aloha-csma-analyzer
   ```
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install pinned dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 10. Running the Application

Launch the Streamlit web application:
```powershell
python -m streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

### App Tabs Overview
- **📊 Dashboard**: KPI performance summary cards (packets generated, successful delivery %, collisions, throughput, delay).
- **📋 Simulation**: Interactive packet-level inspection table with multi-status filtering and column sorting.
- **📈 Timeline**: Plotly Gantt-style activity chart showing successful (green) and colliding (red) transmissions per node.
- **⚔️ Comparison**: Side-by-side performance table for all 3 protocols evaluated on an identical packet arrival trace.
- **📉 Load Analysis**: Multi-run parameter sweep line plots with error bars ($\mu \pm \sigma$):
  1. Offered Load vs Throughput
  2. Offered Load vs Collision Rate
  3. Offered Load vs Average Delay
  4. Offered Load vs Average Retransmissions
  5. Number of Nodes vs Throughput
- **📖 Theory**: Static educational reference with analytical formulas and derivation context.
- **ℹ️ About**: Project scope, simulation boundaries, and technical overview.

---

## 11. Running the Automated Test Suite

Run the full pytest suite:
```powershell
python -m pytest -v
```

### Test Coverage Summary (32 Total Tests)
- `tests/test_aloha.py` (9 tests):
  - Isolated transmission success, forced overlap collision, retransmission triggering, max retries packet dropping, sequential non-overlapping success, Slotted ALOHA boundary alignment, slot collision, and seed reproducibility.
- `tests/test_csma.py` (5 tests):
  - Immediate transmission on idle channel, waiting on busy channel, propagation delay tuning to force collision vs avoidance, waiting nodes collision at channel clear, and empirical throughput superiority over ALOHA.
- `tests/test_metrics.py` (10 tests):
  - Validates each metric formula against a hand-computed 10-packet dataset with known delays, retries, and collisions, plus empty dataset edge cases.
- `tests/test_sweep.py` (4 tests):
  - Verifies comparison mode identical packet arrival timestamps across all 3 protocols, load sweep output structure, hand-computed mean/std calculations, and single-run zero-std handling.
- `tests/test_ui.py` (4 tests):
  - Streamlit AppTest integration tests for startup, Pure ALOHA single simulation, Slotted ALOHA single simulation, CSMA single simulation, Comparison mode, and Load Sweep execution.
