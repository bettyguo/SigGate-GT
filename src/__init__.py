"""
SigGate-GT: Taming Over-Smoothing in Graph Transformers via Sigmoid-Gated Attention.

Element-wise sigmoid gating on the attention output within the GraphGPS framework,
addressing over-smoothing, attention entropy collapse, and training instability.
"""

from siggate_gt.__version__ import __version__
from siggate_gt.models import SigGateGT, SigGateGPSLayer, SigGateMultiHeadAttention


__all__ = [
    "__version__",
    "SigGateGT",
    "SigGateGPSLayer",
    "SigGateMultiHeadAttention",
]
