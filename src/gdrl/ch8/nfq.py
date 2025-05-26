import gc
import random
import tempfile
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch
from IPython.display import HTML

from .vis_utils import collect_env_videos, get_gif_html

ERASE_LINE = "\x1b[2K"
LEAVE_PRINT_EVERY_N_SECS = 10  # seconds

LEAVE_PRINT_EVERY_N_SECS = 60
ERASE_LINE = "\x1b[2K"
EPS = 1e-6


def BEEP():
    """Plays a beep sound using the system bell."""
    print("\a", end="", flush=True)


RESULTS_DIR = Path("..") / "results"


class NFQ:
    def __init__(
        self,
        value_model_fn,
        value_optimizer_fn,
        value_optimizer_lr,
        training_strategy_fn,
        evaluation_strategy_fn,
        batch_size,
        epochs,
    ):
        """
        Initialize the NFQ agent with model, optimizer, strategies, batch size, and epochs.

        Args:
        ----
            value_model_fn: Function to create the value model.
            value_optimizer_fn: Function to create the optimizer.
            value_optimizer_lr: Learning rate for the optimizer.
            training_strategy_fn: Function to create the training strategy.
            evaluation_strategy_fn: Function to create the evaluation strategy.
            batch_size: Batch size for training.
            epochs: Number of epochs per training batch.

        """
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.batch_size = batch_size
        self.epochs = epochs

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        _batch_size = len(is_terminals)

        max_a_q_sp = self.online_model(next_states).detach().max(1)[0].unsqueeze(1)

        target_q_s = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_s
        value_loss = td_errors.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, truncated, info = env.step(action)
        is_truncated = "TimeLimit.truncated" in info and info["TimeLimit.truncated"]
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.experiences.append(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(
            self.training_strategy.exploratory_action_taken
        )
        return new_state, is_terminal

    def train(
        self,
        make_env_fn,
        make_env_kargs,
        seed,
        gamma,
        max_minutes,
        max_episodes,
        goal_mean_100_reward,
    ):
        training_start, last_debug_time = time.time(), float("-inf")

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.online_model = self.value_model_fn(nS, nA)
        self.value_optimizer = self.value_optimizer_fn(
            self.online_model, self.value_optimizer_lr
        )

        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()
        self.experiences = []

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            obs, _info = env.reset()  # Unpack observation and info dictionary
            state = obs  # Assign the observation to the state variable
            is_terminal = False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for _step in count():
                state, is_terminal = self.interaction_step(state, env)

                if len(self.experiences) >= self.batch_size:
                    states, actions, rewards, next_states, is_terminals = zip(
                        *self.experiences, strict=False
                    )
                    states = np.vstack(states)
                    actions = np.vstack(actions)
                    rewards = np.vstack(rewards)
                    next_states = np.vstack(next_states)
                    is_terminals = np.vstack(is_terminals)
                    batches = [states, actions, rewards, next_states, is_terminals]
                    experiences = self.online_model.load(batches)
                    for _ in range(self.epochs):
                        self.optimize_model(experiences)
                    self.experiences.clear()

                if is_terminal:
                    gc.collect()
                    break

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.save_checkpoint(episode - 1, self.online_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.array(
                self.episode_timestep[-100:]
            )
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = (
                total_step,
                mean_100_reward,
                mean_100_eval_score,
                training_time,
                wallclock_elapsed,
            )

            reached_debug_time = (
                time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            )
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = (
                reached_max_minutes or reached_max_episodes or reached_goal_mean_reward
            )

            elapsed_str = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - training_start)
            )
            debug_message = "el {}, ep {:04}, ts {:06}, "
            debug_message += "ar 10 {:05.1f}\u00b1{:05.1f}, "
            debug_message += "100 {:05.1f}\u00b1{:05.1f}, "
            debug_message += "ex 100 {:02.1f}\u00b1{:02.1f}, "
            debug_message += "ev {:05.1f}\u00b1{:05.1f}"
            debug_message = debug_message.format(
                elapsed_str,
                episode - 1,
                total_step,
                mean_10_reward,
                std_10_reward,
                mean_100_reward,
                std_100_reward,
                mean_100_exp_rat,
                std_100_exp_rat,
                mean_100_eval_score,
                std_100_eval_score,
            )
            print(debug_message, end="\r", flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes:
                    print("--> reached_max_minutes \u2715")
                if reached_max_episodes:
                    print("--> reached_max_episodes \u2715")
                if reached_goal_mean_reward:
                    print("--> reached_goal_mean_reward \u2713")
                break

        final_eval_score, score_std = self.evaluate(
            self.online_model, env, n_episodes=100
        )
        wallclock_time = time.time() - training_start
        print("Training complete.")
        print(
            f"Final evaluation score {final_eval_score:.2f}\u00b1{score_std:.2f} in {training_time:.2f}s training time,"
            f" {wallclock_time:.2f}s wall-clock time.\n"
        )
        env.close()
        del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            obs, info = eval_env.reset()  # Unpack observation and info dictionary
            s = obs  # Assign the observation to the state variable
            d = False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _, _ = eval_env.step(a)
                rs[-1] += r
                if d:
                    break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = list(Path(self.checkpoint_dir).glob("*.tar"))
        paths_dic = {int(path.name.split(".")[-2]): path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = (
            np.linspace(1, last_ep + 1, n_checkpoints, endpoint=True, dtype=np.int64)
            - 1
        )

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                Path(path).unlink()

        return self.checkpoint_paths

    def demo_last(self, title="Fully-trained {} Agent", n_episodes=3, max_n_videos=3):
        env = self.make_env_fn(
            **self.make_env_kargs,
            monitor_mode="evaluation",
            render_mode="rgb_array",
            record=True,
        )

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.online_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.online_model, env, n_episodes=n_episodes)
        env_videos = collect_env_videos(env)
        data = get_gif_html(
            env_videos=env_videos,
            title=title.format(self.__class__.__name__),
            max_n_videos=max_n_videos,
        )
        del env
        return HTML(data=data)

    def demo_progression(self, title="{} Agent progression", max_n_videos=5):
        env = self.make_env_fn(
            **self.make_env_kargs,
            monitor_mode="evaluation",
            render_mode="rgb_array",
            record=True,
        )

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.online_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.online_model, env, n_episodes=1)

        env.close()
        env_videos = collect_env_videos(env)
        data = get_gif_html(
            env_videos=env_videos,
            title=title.format(self.__class__.__name__),
            subtitle_eps=checkpoint_paths,
            max_n_videos=max_n_videos,
        )
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(
            model.state_dict(),
            Path(self.checkpoint_dir) / f"model.{episode_idx}.tar",
        )
