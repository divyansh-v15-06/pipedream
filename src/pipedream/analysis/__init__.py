"""Statistical summaries for paired compiler experiments."""

from .stats import BootstrapInterval, paired_bootstrap, paired_effect_size, summarize_differences

__all__ = [
    "BootstrapInterval",
    "paired_bootstrap",
    "paired_effect_size",
    "summarize_differences",
]
