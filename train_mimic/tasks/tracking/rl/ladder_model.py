"""Ladder actor and critic. Every observation feature is continuous."""

from .temporal_cnn_model import TemporalCNNModel


class LadderTemporalCNNModel(TemporalCNNModel):
    """Same temporal CNN as tracking, with no categorical prefix left unchanged."""
