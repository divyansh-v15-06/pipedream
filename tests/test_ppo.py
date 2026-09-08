from pipedream.agents import PPOConfig


def test_ppo_config_checksum_is_stable() -> None:
    config = PPOConfig(total_timesteps=32, rollout_steps=8, batch_size=8)

    assert config.checksum() == config.checksum()
    assert len(config.checksum()) == 64
