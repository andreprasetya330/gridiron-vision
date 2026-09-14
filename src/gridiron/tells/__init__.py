"""Tell mining: what a defense gives away, and whether it survives scrutiny."""

from gridiron.tells.mining import Tell, TellMiner, MiningConfig
from gridiron.tells.stats import (
    benjamini_hochberg,
    beta_binomial_posterior,
    estimate_prior_strength,
    two_proportion_pvalue,
    wilson_interval,
)
from gridiron.tells.validation import ValidationResult, forward_validate

__all__ = [
    "Tell",
    "TellMiner",
    "MiningConfig",
    "ValidationResult",
    "forward_validate",
    "benjamini_hochberg",
    "beta_binomial_posterior",
    "estimate_prior_strength",
    "two_proportion_pvalue",
    "wilson_interval",
]
