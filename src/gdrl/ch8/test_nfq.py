import numpy as np
import pytest
import torch

from .nfq import NFQ


class DummyModel(torch.nn.Module):
    def __init__(self, ns=4, na=2):
        """Initialize the DummyModel with state and action dimensions."""
        super().__init__()
        self.ns = ns
        self.nA = na
        self.linear = torch.nn.Linear(ns, na)

    def forward(self, x):
        # Accepts numpy or torch, returns Q-values
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.linear(x)

    def load(self, batches):
        # Dummy loader for batches
        states, actions, rewards, next_states, is_terminals = batches
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(is_terminals, dtype=torch.float32),
        )


class DummyOptimizer(torch.optim.Adam):
    def __init__(self, params, lr):
        """Initialize the DummyOptimizer with parameters and learning rate."""
        super().__init__(params, lr=lr)


class DummyStrategy:
    def __init__(self):
        """Initialize the DummyStrategy."""
        self.exploratory_action_taken = False

    def select_action(self, _model, _state):
        # Always select action 0
        return 0


class DummyEnv:
    def __init__(self):
        """Initialize the DummyEnv."""
        self.observation_space = type("ObsSpace", (), {"shape": [4]})()
        self.action_space = type(
            "ActSpace", (), {"n": 2, "seed": lambda _self, _s: None}
        )()
        self.reset_called = False
        self.step_called = False
        self._step_count = 0

    def reset(self):
        self.reset_called = True
        return np.zeros(4), {}

    def step(self, _action):
        self.step_called = True
        self._step_count += 1
        # End after 2 steps
        done = self._step_count >= 2  # noqa: PLR2004
        return np.zeros(4), 1.0, done, False, {}

    def close(self):
        """Close the environment (dummy method to avoid errors)."""


@pytest.fixture()
def nfq_agent():
    """Fixture to create a minimal NFQ agent with dummy components."""
    return NFQ(
        value_model_fn=lambda ns, na: DummyModel(ns, na),
        value_optimizer_fn=lambda model, lr: DummyOptimizer(model.parameters(), lr),
        value_optimizer_lr=0.01,
        training_strategy_fn=DummyStrategy,
        evaluation_strategy_fn=DummyStrategy,
        batch_size=2,
        epochs=1,
    )


def test_nfq_init(nfq_agent):
    """Test that the NFQ agent initializes with the correct attributes."""
    assert hasattr(nfq_agent, "value_model_fn")
    assert hasattr(nfq_agent, "value_optimizer_fn")
    assert nfq_agent.batch_size == 2  # noqa: PLR2004


def test_nfq_train_and_evaluate(nfq_agent):
    """Test that the NFQ agent can train and evaluate with dummy environment."""
    # Patch make_env_fn to return DummyEnv
    nfq_agent.make_env_fn = lambda **_kwargs: DummyEnv()
    nfq_agent.make_env_kargs = {}
    nfq_agent.seed = 42
    nfq_agent.gamma = 0.99

    # Patch save_checkpoint to do nothing
    nfq_agent.save_checkpoint = lambda _episode_idx, _model: None
    # Patch get_cleaned_checkpoints to avoid filesystem
    nfq_agent.get_cleaned_checkpoints = lambda _n_checkpoints=5: {0: "dummy_path"}

    # Run train for a few episodes
    result, final_eval_score, training_time, wallclock_time = nfq_agent.train(
        nfq_agent.make_env_fn,
        nfq_agent.make_env_kargs,
        nfq_agent.seed,
        nfq_agent.gamma,
        max_minutes=0.01,
        max_episodes=2,
        goal_mean_100_reward=1.0,
    )
    assert result.shape[0] == 2  # noqa: PLR2004
    assert isinstance(final_eval_score, float)
    assert training_time >= 0
    assert wallclock_time >= 0


def test_nfq_optimize_model(nfq_agent):
    """Test that optimize_model runs without error on dummy data."""
    nfq_agent.online_model = DummyModel()
    nfq_agent.value_optimizer = DummyOptimizer(
        nfq_agent.online_model.parameters(), 0.01
    )
    nfq_agent.gamma = 0.99
    # Create dummy experiences
    batch_size = 2
    states = torch.zeros((batch_size, 4), dtype=torch.float32)
    actions = torch.zeros((batch_size, 1), dtype=torch.int64)
    rewards = torch.ones((batch_size, 1), dtype=torch.float32)
    next_states = torch.zeros((batch_size, 4), dtype=torch.float32)
    is_terminals = torch.zeros((batch_size, 1), dtype=torch.float32)
    nfq_agent.optimize_model((states, actions, rewards, next_states, is_terminals))


def test_get_cleaned_checkpoints_uses_builtin_int(tmp_path):
    """Test that get_cleaned_checkpoints does not use np.int (should use int or np.int64)."""
    agent = NFQ(
        value_model_fn=lambda _ns, _na: DummyModel(),
        value_optimizer_fn=lambda _model, _lr: None,
        value_optimizer_lr=0.01,
        training_strategy_fn=DummyStrategy,
        evaluation_strategy_fn=DummyStrategy,
        batch_size=2,
        epochs=1,
    )
    # Simulate a checkpoint directory with fake .tar files
    agent.checkpoint_dir = tmp_path
    # Create dummy checkpoint files: model.0.tar, model.1.tar, ...
    for i in range(3):
        (tmp_path / f"model.{i}.tar").touch()
    # This should not raise AttributeError about np.int

    agent.get_cleaned_checkpoints(n_checkpoints=2)
