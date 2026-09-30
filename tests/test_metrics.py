"""Unit tests for simulation performance metrics using hand-computed reference values."""

import math
import pytest

from simulation.base import Packet, PacketStatus
from simulation.metrics import (
    avg_delay,
    avg_retransmissions,
    calculate_all_metrics,
    channel_utilization,
    collision_rate,
    dropped_packets,
    normalized_throughput,
    success_rate,
    throughput,
)


@pytest.fixture
def hand_computed_dataset():
    """Provides a deterministic hand-computed dataset with 10 packets and known transmission history.

    Configuration:
        sim_time = 20.0
        packet_duration = 1.0

    Packets (10 total):
        - Packet 0: gen=1.0, succ=2.0, retries=0, status=SUCCESS (delay = 1.0)
        - Packet 1: gen=3.0, succ=4.0, retries=0, status=SUCCESS (delay = 1.0)
        - Packet 2: gen=5.0, succ=6.0, retries=0, status=SUCCESS (delay = 1.0)
        - Packet 3: gen=7.0, succ=8.0, retries=0, status=SUCCESS (delay = 1.0)
        - Packet 4: gen=9.0, succ=10.0, retries=0, status=SUCCESS (delay = 1.0)
        - Packet 5: gen=11.0, succ=13.0, retries=1, status=SUCCESS (delay = 2.0)
        - Packet 6: gen=14.0, succ=17.0, retries=2, status=SUCCESS (delay = 3.0)
        - Packet 7: gen=11.0, succ=None, retries=1, status=DROPPED
        - Packet 8: gen=14.0, succ=None, retries=1, status=DROPPED
        - Packet 9: gen=14.0, succ=None, retries=1, status=DROPPED

    Channel Log (10 transmission attempts):
        1. (1.0, 2.0, 0, "SUCCESS")
        2. (3.0, 4.0, 1, "SUCCESS")
        3. (5.0, 6.0, 2, "SUCCESS")
        4. (7.0, 8.0, 3, "SUCCESS")
        5. (9.0, 10.0, 4, "SUCCESS")
        6. (11.0, 12.0, 5, "COLLISION")
        7. (11.0, 12.0, 7, "COLLISION")
        8. (12.0, 13.0, 5, "SUCCESS")
        9. (14.0, 15.0, 6, "COLLISION")
        10. (16.0, 17.0, 6, "SUCCESS")
        (Total attempts = 10, 7 SUCCESS, 3 COLLISION)

    Expected Metric Values:
        - throughput = 7 / 20.0 = 0.35 packets/sec
        - normalized_throughput = 0.35 * 1.0 = 0.35
        - collision_rate = 3 / 10 = 0.30
        - success_rate = 7 / 10 = 0.70
        - avg_delay = (1.0*5 + 2.0 + 3.0) / 7 = 10.0 / 7 ≈ 1.42857142857
        - avg_retransmissions = (0*5 + 1 + 2 + 1 + 1 + 1) / 10 = 6 / 10 = 0.60
        - dropped_packets = 3
        - channel_utilization:
            Busy intervals: [1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [12,13], [14,15], [16,17]
            Merged [11,13] (2.0s) + seven 1.0s intervals = 9.0s total busy
            utilization = 9.0 / 20.0 = 0.45
    """
    sim_time = 20.0
    packet_duration = 1.0

    packets = [
        Packet(packet_id=0, source_node=0, generation_time=1.0, attempt_time=1.0, successful_time=2.0, retry_count=0, status=PacketStatus.SUCCESS),
        Packet(packet_id=1, source_node=1, generation_time=3.0, attempt_time=3.0, successful_time=4.0, retry_count=0, status=PacketStatus.SUCCESS),
        Packet(packet_id=2, source_node=2, generation_time=5.0, attempt_time=5.0, successful_time=6.0, retry_count=0, status=PacketStatus.SUCCESS),
        Packet(packet_id=3, source_node=3, generation_time=7.0, attempt_time=7.0, successful_time=8.0, retry_count=0, status=PacketStatus.SUCCESS),
        Packet(packet_id=4, source_node=4, generation_time=9.0, attempt_time=9.0, successful_time=10.0, retry_count=0, status=PacketStatus.SUCCESS),
        Packet(packet_id=5, source_node=5, generation_time=11.0, attempt_time=12.0, successful_time=13.0, retry_count=1, status=PacketStatus.SUCCESS),
        Packet(packet_id=6, source_node=6, generation_time=14.0, attempt_time=16.0, successful_time=17.0, retry_count=2, status=PacketStatus.SUCCESS),
        Packet(packet_id=7, source_node=7, generation_time=11.0, attempt_time=11.0, successful_time=None, retry_count=1, status=PacketStatus.DROPPED),
        Packet(packet_id=8, source_node=8, generation_time=14.0, attempt_time=14.0, successful_time=None, retry_count=1, status=PacketStatus.DROPPED),
        Packet(packet_id=9, source_node=9, generation_time=14.0, attempt_time=14.0, successful_time=None, retry_count=1, status=PacketStatus.DROPPED),
    ]

    channel_log = [
        (1.0, 2.0, 0, "SUCCESS"),
        (3.0, 4.0, 1, "SUCCESS"),
        (5.0, 6.0, 2, "SUCCESS"),
        (7.0, 8.0, 3, "SUCCESS"),
        (9.0, 10.0, 4, "SUCCESS"),
        (11.0, 12.0, 5, "COLLISION"),
        (11.0, 12.0, 7, "COLLISION"),
        (12.0, 13.0, 5, "SUCCESS"),
        (14.0, 15.0, 6, "COLLISION"),
        (16.0, 17.0, 6, "SUCCESS"),
    ]

    return {
        "packets": packets,
        "channel_log": channel_log,
        "sim_time": sim_time,
        "packet_duration": packet_duration,
    }


def test_throughput_formula(hand_computed_dataset):
    d = hand_computed_dataset
    tp = throughput(d["packets"], d["channel_log"], d["sim_time"])
    assert math.isclose(tp, 0.35, abs_tol=1e-9)


def test_normalized_throughput_formula(hand_computed_dataset):
    d = hand_computed_dataset
    norm_tp = normalized_throughput(
        d["packets"], d["channel_log"], d["sim_time"], packet_duration=d["packet_duration"]
    )
    assert math.isclose(norm_tp, 0.35, abs_tol=1e-9)


def test_collision_rate_formula(hand_computed_dataset):
    d = hand_computed_dataset
    col_rate = collision_rate(d["packets"], d["channel_log"], d["sim_time"])
    assert math.isclose(col_rate, 0.30, abs_tol=1e-9)


def test_success_rate_formula(hand_computed_dataset):
    d = hand_computed_dataset
    succ_rate = success_rate(d["packets"], d["channel_log"], d["sim_time"])
    assert math.isclose(succ_rate, 0.70, abs_tol=1e-9)


def test_avg_delay_formula(hand_computed_dataset):
    d = hand_computed_dataset
    delay = avg_delay(d["packets"], d["channel_log"], d["sim_time"])
    expected = 10.0 / 7.0
    assert math.isclose(delay, expected, abs_tol=1e-9)


def test_avg_retransmissions_formula(hand_computed_dataset):
    d = hand_computed_dataset
    avg_retries = avg_retransmissions(d["packets"], d["channel_log"], d["sim_time"])
    assert math.isclose(avg_retries, 0.60, abs_tol=1e-9)


def test_dropped_packets_formula(hand_computed_dataset):
    d = hand_computed_dataset
    dropped = dropped_packets(d["packets"], d["channel_log"], d["sim_time"])
    assert dropped == 3


def test_channel_utilization_formula(hand_computed_dataset):
    d = hand_computed_dataset
    util = channel_utilization(d["packets"], d["channel_log"], d["sim_time"])
    assert math.isclose(util, 0.45, abs_tol=1e-9)


def test_calculate_all_metrics_dict(hand_computed_dataset):
    d = hand_computed_dataset
    all_metrics = calculate_all_metrics(
        d["packets"], d["channel_log"], d["sim_time"], packet_duration=d["packet_duration"]
    )
    assert math.isclose(all_metrics["throughput"], 0.35, abs_tol=1e-9)
    assert math.isclose(all_metrics["normalized_throughput"], 0.35, abs_tol=1e-9)
    assert math.isclose(all_metrics["collision_rate"], 0.30, abs_tol=1e-9)
    assert math.isclose(all_metrics["success_rate"], 0.70, abs_tol=1e-9)
    assert math.isclose(all_metrics["avg_delay"], 10.0 / 7.0, abs_tol=1e-9)
    assert math.isclose(all_metrics["avg_retransmissions"], 0.60, abs_tol=1e-9)
    assert all_metrics["dropped_packets"] == 3
    assert math.isclose(all_metrics["channel_utilization"], 0.45, abs_tol=1e-9)


def test_metrics_empty_dataset_edge_cases():
    """Verify that all metrics gracefully return 0.0 or 0 on empty inputs or non-positive sim_time."""
    assert throughput([], [], 0.0) == 0.0
    assert normalized_throughput([], [], 0.0) == 0.0
    assert collision_rate([], [], 10.0) == 0.0
    assert success_rate([], [], 10.0) == 0.0
    assert avg_delay([], [], 10.0) == 0.0
    assert avg_retransmissions([], [], 10.0) == 0.0
    assert dropped_packets([], [], 10.0) == 0
    assert channel_utilization([], [], 10.0) == 0.0
