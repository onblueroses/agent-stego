import pytest

from experiments import conversation_detection, detectability_scaling, latent_bandwidth


@pytest.mark.parametrize(
    "module",
    [conversation_detection, detectability_scaling, latent_bandwidth],
)
def test_invalidated_experiment_exits_before_work(module) -> None:
    with pytest.raises(SystemExit, match="retired"):
        module.main()
