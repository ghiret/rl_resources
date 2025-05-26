import base64
import json
import subprocess
from pathlib import Path

import numpy as np


def get_videos_html(env_videos, title, max_n_videos=5):
    """
    Generate HTML to display embedded MP4 videos with episode metadata.

    Args:
    ----
        env_videos (list): List of tuples containing video and metadata file paths.
        title (str): Title for the HTML section.
        max_n_videos (int): Maximum number of videos to display.

    Returns:
    -------
        str or None: HTML string for displaying videos, or None if no videos are provided.

    """
    videos = np.array(env_videos)
    if len(videos) == 0:
        return None

    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = (
        np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1]
    )
    videos = videos[idxs, ...]

    strm = f"<h2>{title}<h2>"
    for video_path, meta_path in videos:
        with Path(video_path).open("r+b") as video_file:
            video = video_file.read()

            encoded = base64.b64encode(video)

            with Path(meta_path).open() as data_file:
                meta = json.load(data_file)

            html_tag = """
            <h3>{0}<h3/>
            <video width="960" height="540" controls>
                <source src="data:video/mp4;base64,{1}" type="video/mp4" />
            </video>"""
            strm += html_tag.format(
                "Episode " + str(meta["episode_id"]), encoded.decode("ascii")
            )
    return strm


def extract_episode_id(video_path):
    """
    Extract the episode number from a video filename like '.../episode-0.mp4'.

    Args:
    ----
        video_path (str or Path): Path to the video file.
    Returns the episode number as a string (or int if you prefer).

    """
    stem = Path(video_path).stem  # e.g., 'episode-0'
    if "-" in stem:
        # Split on '-' and take the last part
        return stem.split("-")[-1]
    return stem  # fallback: return the whole stem if no '-'


def get_gif_html(env_videos, title, subtitle_eps=None, max_n_videos=4):
    """
    Generate HTML to display GIFs converted from video files, with optional subtitles and a maximum number of videos.

    Args:
    ----
        env_videos (list): List of tuples containing video and metadata file paths.
        title (str): Title for the HTML section.
        subtitle_eps (dict or None): Optional mapping for episode subtitles.
        max_n_videos (int): Maximum number of videos to display.

    Returns:
    -------
        str or None: HTML string for displaying GIFs, or None if no videos are provided.

    """
    videos = np.array(env_videos)
    if len(videos) == 0:
        return None

    n_videos = max(1, min(max_n_videos, len(videos)))
    idxs = (
        np.linspace(0, len(videos) - 1, n_videos).astype(int) if n_videos > 1 else [-1]
    )
    videos = videos[idxs, ...]

    strm = f"<h2>{title}<h2>"
    for video_path, meta_path in videos:
        basename = Path(video_path).with_suffix("")
        gif_path = str(basename) + ".gif"
        if not Path(gif_path).exists():
            import shutil

            ffmpeg_path = shutil.which("ffmpeg")
            convert_path = shutil.which("convert")
            if ffmpeg_path is None or convert_path is None:
                error_msg = "ffmpeg or convert not found in PATH"
                raise RuntimeError(error_msg)

            ps = subprocess.Popen(
                (
                    ffmpeg_path,
                    "-i",
                    video_path,
                    "-r",
                    "7",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "ppm",
                    "-crf",
                    "20",
                    "-vf",
                    "scale=512:-1",
                    "-",
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # Ensure gif_path is a safe string and not user-controlled
            safe_gif_path = Path(gif_path).resolve()
            _output = subprocess.check_output(
                [
                    convert_path,
                    "-coalesce",
                    "-delay",
                    "7",
                    "-loop",
                    "0",
                    "-fuzz",
                    "2%",
                    "+dither",
                    "-deconstruct",
                    "-layers",
                    "Optimize",
                    "-",
                    str(safe_gif_path),
                ],
                stdin=ps.stdout,
                stderr=subprocess.DEVNULL,
            )
            ps.wait()

        with Path(gif_path).open("r+b") as gif_file:
            gif = gif_file.read()
            encoded = base64.b64encode(gif)

        # Handle missing meta_path gracefully
        if meta_path is not None and Path(meta_path).exists():
            with Path(meta_path).open() as data_file:
                meta = json.load(data_file)
            episode_id = meta.get("episode_id", Path(video_path).stem)
        else:
            episode_id = extract_episode_id(video_path)

        html_tag = """
        <h3>{0}<h3/>
        <img src="data:image/gif;base64,{1}" />"""
        prefix = "Trial " if subtitle_eps is None else "Episode "
        # print(f"video_path: {video_path}")
        # print(f"Episode ID: {episode_id}")
        # print(f"Subtitle EPS: {subtitle_eps}")
        if subtitle_eps is None:
            sufix = str(episode_id)
        else:
            sufix = str(subtitle_eps.get(int(episode_id), episode_id))

        strm += html_tag.format(prefix + sufix, encoded.decode("ascii"))
    return strm


def collect_env_videos(env):
    """
    Collect video file paths from the RecordVideo wrapper's video folder.

    Returns a list of (video_path, None) tuples for get_gif_html.
    """
    video_folder = getattr(env, "video_folder", None)
    if video_folder is not None:
        video_paths = sorted(Path(video_folder).glob("*.mp4"))
        return [(str(path), None) for path in video_paths]
    return []
