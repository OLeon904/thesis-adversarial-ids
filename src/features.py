"""
CICFlowMeter feature typing for constraint-aware adversarial attacks.

Locked thesis decisions:
- Destination Port, Protocol, TCP window init fields, header lengths: IMMUTABLE
- Count-like fields: DISCRETE INTEGER (project after perturbation)
- Timing/rate/statistics: CONTINUOUS (primary perturbable space)
- Flow Duration: coherence-controlled (not freely perturbed in attacks)
"""

from __future__ import annotations

# Columns to drop before modeling (identifiers / leakage)
LEAKAGE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Timestamp",
]

# Immutable under adversarial perturbation (mask gradient / zero delta)
IMMUTABLE_FEATURES = [
    "Destination Port",
    "Protocol",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Bwd PSH Flags",
    "Bwd URG Flags",
]

# Derived timing anchor — perturbed only via projection, not free PGD step
COHERENCE_ANCHOR_FEATURES = [
    "Flow Duration",
]

# Integer-domain count features
DISCRETE_INTEGER_FEATURES = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "act_data_pkt_fwd",
]

# Primary continuous perturbable features (attack-eligible subset)
CONTINUOUS_FEATURES = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

IAT_FEATURES = [
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
]


def classify_columns(feature_names: list[str]) -> dict[str, list[str]]:
    """Map available columns to constraint groups (intersection with dataset)."""
    avail = set(feature_names)

    def pick(names: list[str]) -> list[str]:
        return [c for c in names if c in avail]

    immutable = pick(IMMUTABLE_FEATURES)
    coherence = pick(COHERENCE_ANCHOR_FEATURES)
    discrete = pick(DISCRETE_INTEGER_FEATURES)
    continuous = pick(CONTINUOUS_FEATURES)
    # Any remaining numeric columns (e.g. bulk stats) default to continuous
    assigned = set(immutable + coherence + discrete + continuous)
    other = sorted(avail - assigned)
    continuous = continuous + other

    return {
        "immutable": immutable,
        "coherence_anchor": coherence,
        "discrete_integer": discrete,
        "continuous": continuous,
        "iat": pick(IAT_FEATURES),
    }
