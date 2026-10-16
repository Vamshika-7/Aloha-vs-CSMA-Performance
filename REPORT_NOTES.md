# Simulation Empirical Report Notes

> **Disclaimer**: This document contains empirical measurements obtained directly from fresh runs of the discrete-event simulation engine in `aloha-csma-analyzer`. This project is an educational, mathematical discrete-event simulation testbed, not a physical packet transmission system.

---

## 1. Default Parameters Configuration

The baseline evaluation was executed under the following standard configuration:

| Parameter | Symbol / Variable | Value | Description |
| :--- | :--- | :--- | :--- |
| Contending Nodes | $N$ (`num_nodes`) | **10** | Contending independent stations sharing the channel |
| Simulation Duration | $T_{\text{sim}}$ (`sim_time`) | **1000.0 s** | Total discrete-event simulation window |
| Offered Load | $G$ (`offered_load`) | **0.10** | Newly generated Poisson arrival rate ($\lambda_{\text{new}} \cdot T = 0.10$ pkts/s) |
| Packet Duration | $T$ (`packet_duration`) | **1.0 s** | Transmission airtime per packet |
| Propagation Delay | $\tau$ (`propagation_delay`) | **0.01 s** | Physical one-way medium transit time ($a = \tau / T = 0.01$) |
| Maximum Retries | $k_{\max}$ (`max_retries`) | **10** | Maximum retry attempts before dropping a packet |
| Random Seed | `seed` | **42** | Deterministic SeedSequence base for reproducible streams |

---

## 2. Default Parameter Protocol Results (Fresh Run)

The three protocols were evaluated on an **identical packet arrival trace** generated under `seed=42`:

| Metric | Pure ALOHA | Slotted ALOHA | 1-Persistent CSMA | Unit / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Packets Generated** | 104 | 104 | 104 | Total packets generated across all 10 nodes |
| **Successful Packets** | 104 | 104 | 104 | Successfully delivered packets |
| **Collisions** | 55 | 17 | 0 | Collided transmission attempts (`collision_attempts`) |
| **Dropped Packets** | 0 | 0 | 0 | Packets exceeding 10 retries |
| **Throughput** | 0.1040 | 0.1040 | 0.1040 | Delivered packets per second ($\text{pkts}/s$) |
| **Collision Rate** | 34.59% | 14.05% | 0.00% | $\text{collisions} / \text{total attempts}$ |
| **Average Delay** | 3.7603 | 1.8655 | 1.0433 | Mean latency from generation to delivery ($s$) |
| **Average Retransmissions** | 0.5288 | 0.1635 | 0.0000 | Mean retries per generated packet |
| **Channel Utilization** | 10.40% | 10.40% | 10.40% | Useful successful airtime fraction ($\frac{104 \times 1.0}{1000}$) |
| **Channel Busy Fraction** | 14.07% | 11.20% | 10.40% | Total channel occupancy including collisions |

---

## 3. Parametric Load Sweep ($N=10$, Range $G \in [0.05, 1.00]$)

A baseline offered load sweep was executed across 20 points ($G \in [0.05, 1.00]$ in steps of $0.05$) with $N = 10$ nodes and 5 independent runs per point ($T_{\text{sim}} = 1000\,\text{s}$):

### Mean Throughput per Protocol across Offered Loads

| Offered Load ($G$) | 1-Persistent CSMA ($\text{pkts}/s$) | Pure ALOHA ($\text{pkts}/s$) | Slotted ALOHA ($\text{pkts}/s$) |
| :--- | :--- | :--- | :--- |
| **0.05** | 0.0528 | 0.0526 | 0.0526 |
| **0.10** | 0.1090 | 0.1082 | 0.1088 |
| **0.15** | 0.1630 | 0.1584 | 0.1622 |
| **0.20** | 0.2070 | 0.1952 | 0.2060 |
| **0.25** | 0.2584 | 0.2232 | 0.2534 |
| **0.30** | 0.3060 | 0.2430 | 0.2928 |
| **0.35** | 0.3562 | 0.2592 | 0.3384 |
| **0.40** | 0.4132 | 0.2690 | 0.3652 |
| **0.45** | 0.4528 | 0.2920 | 0.3888 |
| **0.50** | 0.5062 | 0.2988 | 0.4020 |
| **0.55** | 0.5494 | 0.3092 | 0.4136 |
| **0.60** | 0.5876 | 0.3264 | 0.4336 |
| **0.65** | 0.6258 | 0.3166 | 0.4408 |
| **0.70** | 0.6554 | 0.3296 | 0.4346 |
| **0.75** | 0.6962 | 0.3448 | 0.4520 |
| **0.80** | 0.6956 | 0.3602 | 0.4844 |
| **0.85** | 0.7262 | 0.3626 | 0.4756 |
| **0.90** | 0.7632 | 0.3638 | 0.4870 |
| **0.95** | 0.7698 | 0.3662 | 0.4914 |
| **1.00** | 0.7878 | 0.3780 | 0.5088 |

> **Boundary Value Note**: In this sweep with $N=10$, the highest throughput occurs at $G = 1.00$ simply because $G=1.00$ was the rightmost boundary of the evaluated sweep range. **The last point of a sweep is not an interior peak.** With only 10 nodes, per-node queues and exponential backoff prevent immediate channel collapse, causing throughput to continue climbing beyond $G = 1.00$. To locate genuine interior peaks and validate against infinite-population theory, a wide sweep ($G \in [0.1, 3.0]$) with large node populations ($N = 100$) is required (see Section 4).

---

## 4. Theory Validation: Infinite-Population Experiment (`scripts/validate_theory.py`)

To rigorously benchmark the discrete-event simulator against analytical random access theory, an experiment was executed across an extended offered load sweep ($G \in [0.1, 3.0]$ in steps of $0.1$, 5 runs per point, $T_{\text{sim}} = 1000\,\text{s}$), systematically scaling node population ($N \in \{10, 30, 100\}$) and retry limits ($k_{\max} \in \{10, 50\}$).

### Peak Throughput and Interior Point Summary

| Protocol | Nodes ($N$) | Max Retries ($k_{\max}$) | Simulated Peak $S$ | Simulated Peak $G^*_{\text{sim}}$ | Theoretical Peak $S_{\max}$ | Theoretical Peak $G^*_{\text{theory}}$ | Interior Peak? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure ALOHA** | **100** | **10** | **0.1660** | **0.20** | $1 / (2e) \approx 0.184$ | $0.5$ | **YES (Interior)** |
| **Pure ALOHA** | **100** | **50** | **0.1710** | **0.20** | $1 / (2e) \approx 0.184$ | $0.5$ | **YES (Interior)** |
| **Slotted ALOHA** | **100** | **10** | **0.3322** | **0.40** | $1 / e \approx 0.368$ | $1.0$ | **YES (Interior)** |
| **Slotted ALOHA** | **100** | **50** | **0.3270** | **0.40** | $1 / e \approx 0.368$ | $1.0$ | **YES (Interior)** |
| Pure ALOHA | 30 | 10 | 0.2454 | 3.00 | $1 / (2e) \approx 0.184$ | $0.5$ | NO (Boundary) |
| Slotted ALOHA | 30 | 10 | 0.5110 | 3.00 | $1 / e \approx 0.368$ | $1.0$ | NO (Boundary) |
| Pure ALOHA | 10 | 10 | 0.5222 | 3.00 | $1 / (2e) \approx 0.184$ | $0.5$ | NO (Boundary) |
| Slotted ALOHA | 10 | 10 | 0.7218 | 2.70 | $1 / e \approx 0.368$ | $1.0$ | YES (Interior) |

---

### Detailed Throughput vs. Load Comparison ($N=100$, $k_{\max}=10$)

| Offered Load ($G$) | Pure ALOHA (Simulated) | Pure ALOHA (Theory: $G e^{-2G}$) | Slotted ALOHA (Simulated) | Slotted ALOHA (Theory: $G e^{-G}$) |
| :--- | :--- | :--- | :--- | :--- |
| **0.1** | 0.1074 | 0.0819 | 0.1076 | 0.0905 |
| **0.2** | **0.1660 (Peak)** | 0.1341 | 0.2040 | 0.1637 |
| **0.3** | 0.1116 | 0.1646 | 0.3000 | 0.2222 |
| **0.4** | 0.0912 | 0.1797 | **0.3322 (Peak)** | 0.2681 |
| **0.5** | 0.0710 | **0.1839 (Max)** | 0.3064 | 0.3033 |
| **0.6** | 0.0578 | 0.1807 | 0.2808 | 0.3293 |
| **0.7** | 0.0580 | 0.1726 | 0.2744 | 0.3476 |
| **0.8** | 0.0640 | 0.1615 | 0.2778 | 0.3595 |
| **0.9** | 0.0574 | 0.1488 | 0.2672 | 0.3659 |
| **1.0** | 0.0612 | 0.1353 | 0.2658 | **0.3679 (Max)** |
| **1.1** | 0.0586 | 0.1219 | 0.2666 | 0.3662 |
| **1.2** | 0.0554 | 0.1089 | 0.2612 | 0.3614 |
| **1.3** | 0.0598 | 0.0966 | 0.2604 | 0.3543 |
| **1.4** | 0.0574 | 0.0851 | 0.2596 | 0.3452 |
| **1.5** | 0.0588 | 0.0747 | 0.2664 | 0.3347 |
| **1.6** | 0.0570 | 0.0652 | 0.2702 | 0.3230 |
| **1.7** | 0.0554 | 0.0567 | 0.2712 | 0.3106 |
| **1.8** | 0.0600 | 0.0492 | 0.2576 | 0.2975 |
| **1.9** | 0.0530 | 0.0425 | 0.2686 | 0.2842 |
| **2.0** | 0.0584 | 0.0366 | 0.2762 | 0.2707 |
| **2.5** | 0.0564 | 0.0168 | 0.2766 | 0.2052 |
| **3.0** | 0.0638 | 0.0074 | 0.2734 | 0.1494 |

---

### Empirical Findings: Does the Simulator Approach Theory?

**Yes, the discrete-event simulator closely approaches classical theoretical limits as node population $N$ grows:**

1. **Convergence Toward Theoretical Maximum Throughput**:
   - For **Pure ALOHA**, increasing the node count to $N = 100$ brings the simulated peak throughput to **0.1660** (with $k_{\max}=10$) and **0.1710** (with $k_{\max}=50$). These values represent **$90.3\%$ and $93.0\%$** of the theoretical infinite-population maximum ($S_{\max} = 1/(2e) \approx 0.1839$).
   - For **Slotted ALOHA**, at $N = 100$, the simulated peak throughput reaches **0.3322** (with $k_{\max}=10$) and **0.3270** (with $k_{\max}=50$). These values represent **$90.3\%$ and $88.9\%$** of the theoretical infinite-population maximum ($S_{\max} = 1/e \approx 0.3679$).
   - In contrast, with small populations ($N = 10$ and $N = 30$), finite queues throttle retransmissions and prevent collision cascades, causing simulated throughput to stay inflated and peak only at the boundary ($G = 3.0$).

2. **Presence of Genuine Interior Peaks and Channel Collapse**:
   - At $N = 100$, both protocols exhibit clear, unambiguous **interior peaks**:
     - Pure ALOHA peaks at $G^* = 0.20$, collapsing rapidly to $\approx 0.057$ by $G = 0.6$.
     - Slotted ALOHA peaks at $G^* = 0.40$, declining smoothly and stabilizing around $\approx 0.26 - 0.27$ at high loads.
   - The collapse past the interior peak confirms the classical random-access instability predicted by Abramson and Roberts.

3. **Distinction in the Definition of Offered Load ($G$)**:
   - **Simulator Definition**: $G$ is the rate of **newly generated Poisson packet arrivals** per packet duration ($\lambda_{\text{new}} \cdot T$).
   - **Analytical Theory Definition**: In the classical formulas ($S = G e^{-2G}$ and $S = G e^{-G}$), $G$ represents the **total channel attempt rate** (newly generated packets plus all retransmissions) per packet duration ($G_{\text{total}} = \lambda_{\text{total}} \cdot T$).
   - In the simulation, when collisions occur, retransmitted packets are added to the channel alongside new arrivals. Therefore, the actual channel traffic $G_{\text{total}} > G_{\text{new}}$. The channel reaches saturation ($G_{\text{total}} \approx 0.5$ for Pure ALOHA and $G_{\text{total}} \approx 1.0$ for Slotted ALOHA) when newly offered load is around $G_{\text{new}} \approx 0.20$ and $G_{\text{new}} \approx 0.40$. This explains why empirical peaks occur at lower values of newly offered load than the theoretical $0.5$ and $1.0$ markers.

---

## 5. Observations

The empirical simulation data demonstrates the following specific behaviors:

1. **Throughput Equivalence at Low Offered Load**:
   At $G = 0.05$ and $G = 0.10$, all three protocols achieve nearly identical throughput ($\approx 0.053\,\text{pkts}/s$ and $\approx 0.109\,\text{pkts}/s$). When packet arrivals are sparse, transmission overlap probabilities are negligible, and carrier sensing or slotting overhead offers minimal throughput differentiation.

2. **Divergence of Collision Rates and Retransmissions**:
   Even at low load ($G = 0.10$), collision rates and retry burdens diverge markedly:
   - Pure ALOHA exhibits a 34.59% collision rate and 0.5288 average retransmissions per packet.
   - Slotted ALOHA reduces the collision rate to 14.05% and retransmissions to 0.1635.
   - 1-Persistent CSMA registers 0.00% collisions and 0.0000 retransmissions.

3. **Average Latency Profile**:
   Average packet delay directly reflects the retransmission burden:
   - 1-Persistent CSMA delivers packets with a mean latency of $1.0433\,\text{s}$ (close to the single-packet transmission duration of $1.0\,\text{s}$ plus propagation delay).
   - Slotted ALOHA incurs $1.8655\,\text{s}$ due to slot alignment waiting and occasional backoffs.
   - Pure ALOHA averages $3.7603\,\text{s}$ as colliding packets experience repeated exponential backoffs.

4. **Channel Utilization vs. Channel Busy Fraction**:
   In the default comparison run, all three protocols delivered exactly 104 packets of duration $1.0\,\text{s}$ across $1000\,\text{s}$, producing an identical **Channel Utilization of 10.40%** (useful airtime). However, **Channel Busy Fraction** (total occupied airtime) was 14.07% for Pure ALOHA and 11.20% for Slotted ALOHA, confirming that collision airtime accounts for the additional 3.67% and 0.80% channel occupancy. For CSMA, Channel Busy Fraction was exactly 10.40% because zero collisions occurred.

5. **Impact of Contending Node Scaling ($N$)**:
   When $N$ is small ($N=10$), single-node queue restrictions act as implicit flow control: a node waiting in backoff cannot transmit another packet, preventing catastrophic channel saturation. When $N$ scales to 100, independent concurrent transmissions increase, driving the channel into genuine random-access collapse matching theoretical models.

---

## 6. Limitations

The empirical findings documented above are subject to the following structural and modeling boundaries:

1. **Finite Node Population**: Experiments use finite discrete populations ($N = 10, 30, 100$). While $N=100$ closely approximates infinite-population dynamics, residual queueing effects remain present.
2. **Fixed Packet Duration**: All packets are modeled with a uniform transmission duration ($T = 1.0\,\text{s}$). Variable-length packet distributions (e.g., bimodal or Pareto distributions) will alter vulnerability windows and collision durations.
3. **Exponential Backoff Model**: Backoff delays are drawn from a uniform distribution bounded by binary exponential scaling ($U(0, 2^{\min(k, 10)} \cdot T)$). Alternative algorithms (such as p-persistent backoff, linear backoff, or truncated IEEE 802.3 backoff) will produce different latency and throughput dynamics.
4. **Idealized Physical Channel**: The broadcast medium is assumed to be noiseless, error-free, and collision-deterministic (any temporal overlap causes $100\%$ packet loss with no capture effect or physical power capture).
5. **Simulation vs. Real Transmission System**: This engine operates as a discrete-event mathematical simulator using timestamped event queues (`heapq`). It does not execute physical RF modulation, clock drift, hardware transceiver turnaround times, preamble detection, or physical-layer framing overhead.
