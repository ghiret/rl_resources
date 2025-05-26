import logging
import tempfile
from collections.abc import Callable
from typing import Any

import gymnasium as gym
from gymnasium import wrappers

logging.basicConfig(level=logging.INFO)


def get_make_env_fn(**kargs: dict[str, Any]):
    """
    Return a function that creates and configures a Gymnasium environment with optional wrappers and settings.

    Args:
    ----
        **kargs: Additional keyword arguments to pass to the environment constructor.

    Returns:
    -------
        tuple: (make_env_fn, kargs) where make_env_fn is a function to create the environment.

    """

    def make_env_fn(
        env_name: str,  # Expect env_name to be passed when make_env_fn is called
        seed: int | None = None,
        render_mode: str | None = None,
        record: bool = False,  # This will control video recording
        unwrapped: bool = False,
        monitor_mode: str | None = None,  # Used to gate monitoring (stats and/or video)
        inner_wrappers: list[Callable] | None = None,
        outer_wrappers: list[Callable] | None = None,
    ) -> gym.Env:
        env = None

        if render_mode:
            try:
                env = gym.make(env_name, render_mode=render_mode)
                print(
                    f"Created environment '{env_name}' with render mode: {render_mode}"
                )
            except Exception as e:
                logging.warning(f"Could not create env with render={render_mode}: {e}")

        if env is None:
            env = gym.make(env_name)
        print(f"render_mode: {env.render_mode}")
        if seed is not None:
            env.action_space.seed(seed)
        env = env.unwrapped if unwrapped else env

        if inner_wrappers:
            for wrapper in inner_wrappers:
                env = wrapper(env)
        apply_monitoring_wrappers = monitor_mode is not None

        if apply_monitoring_wrappers:
            # Add RecordEpisodeStatistics if monitoring is active, as Monitor used to provide stats.
            # This wrapper collects return, length, and duration of episodes.
            env = wrappers.RecordEpisodeStatistics(env)

            if (
                record
            ):  # 'record' parameter from make_env_fn now directly controls video
                video_output_folder = tempfile.mkdtemp(prefix=f"gym-video-{env_name}-")
                logging.info(f"Recording videos to: {video_output_folder}")
                env = wrappers.RecordVideo(
                    env,
                    video_folder=video_output_folder,
                    # If 'record' is True, this means record the current episode.
                    # For RecordVideo, episode_trigger determines this for each episode.
                    # If 'record=True' for this make_env_fn call means "record all episodes created by this instance":
                    episode_trigger=lambda _episode_id: True,
                    name_prefix=f"rl-video-{env_name}",
                )

        if outer_wrappers:
            for wrapper in outer_wrappers:
                env = wrapper(env)
        return env

    return make_env_fn, kargs
