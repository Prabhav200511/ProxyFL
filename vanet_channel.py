"""Measurement-only VANET wireless link-capacity model."""

from dataclasses import dataclass
import math

from config import (
    VANET_BANDWIDTH_HZ,
    VANET_NOISE_FIGURE_DB,
    VANET_PATH_LOSS_1M_DB,
    VANET_PATH_LOSS_EXPONENT,
    VANET_PHY_MAX_RATE_BPS,
    VANET_TX_POWER_DBM,
)


@dataclass(frozen=True)
class WirelessLink:
    """Description of one successful modeled wireless hop."""

    kind: str
    distance_m: float

    def __post_init__(self):
        if self.kind not in {"v2v", "v2rsu", "rsu2v"}:
            raise ValueError("invalid VANET wireless link kind")
        if not math.isfinite(self.distance_m) or self.distance_m < 0:
            raise ValueError(
                "wireless distance must be finite and non-negative")


def link_capacity_bps(distance_m: float) -> float:
    """Return capped Shannon capacity for the configured VANET link budget."""
    distance = max(float(distance_m), 1.0)
    path_loss = (
        VANET_PATH_LOSS_1M_DB
        + 10.0 * VANET_PATH_LOSS_EXPONENT * math.log10(distance)
    )
    received_dbm = VANET_TX_POWER_DBM - path_loss
    noise_dbm = (
        -174.0
        + 10.0 * math.log10(VANET_BANDWIDTH_HZ)
        + VANET_NOISE_FIGURE_DB
    )
    snr_linear = 10.0 ** ((received_dbm - noise_dbm) / 10.0)
    shannon = VANET_BANDWIDTH_HZ * math.log2(1.0 + snr_linear)
    return min(VANET_PHY_MAX_RATE_BPS, max(shannon, 1.0))


def modeled_airtime_seconds(num_bits: int, capacity_bps: float) -> float:
    """Return observational airtime; it is never applied as a real delay."""
    if num_bits < 0:
        raise ValueError("bit count must be non-negative")
    if not math.isfinite(capacity_bps) or capacity_bps <= 0:
        raise ValueError("capacity must be finite and positive")
    return num_bits / capacity_bps
