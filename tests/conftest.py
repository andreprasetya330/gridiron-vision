import pytest

from gridiron.cues.vocabulary import cues_to_frame
from gridiron.tracking.synthetic import SyntheticConfig, generate_season


@pytest.fixture(scope="session")
def synthetic_config() -> SyntheticConfig:
    # Enough volume for the mining thresholds to be reachable, small enough that
    # the suite stays quick.
    return SyntheticConfig(weeks=10, plays_per_game=60, seed=7)


@pytest.fixture(scope="session")
def synthetic_plays(synthetic_config):
    return list(generate_season(synthetic_config))


@pytest.fixture(scope="session")
def synthetic_cues(synthetic_plays):
    return cues_to_frame(synthetic_plays)
