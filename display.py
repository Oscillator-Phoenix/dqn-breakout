from collections import deque

import os
import shutil

import torch

from utils_env import MyEnv
from utils_drl import Agent

target = 8
model_name = f"model_{target:03d}"
model_path = f"./models/{model_name}"

tmp_frames_dir = "tmp_frames"
movie_dir = "movie"


def render() -> None:

    device = torch.device("cpu")
    env = MyEnv(device)
    agent = Agent(env.get_action_dim(), device, 0.99, 0, 0, 0, 1, model_path)

    obs_queue = deque(maxlen=5)
    avg_reward, frames = env.evaluate(obs_queue, agent, render=True)
    print(f"Avg. Reward: {avg_reward:.1f}")

    if os.path.exists(tmp_frames_dir):
        shutil.rmtree(tmp_frames_dir)
    os.mkdir(tmp_frames_dir)

    if not os.path.exists(movie_dir):
        os.mkdir(movie_dir)

    for ind, frame in enumerate(frames):
        frame.save(os.path.join(tmp_frames_dir,
                                f"{ind:06d}.png"), format="png")

    if os.system(
        f'''ffmpeg -i "./{tmp_frames_dir}/%06d.png" -pix_fmt yuv420p -y ./{movie_dir}/movie-{target:03d}.mp4'''
    ) != 0:
        print("ffmpeg error")


if __name__ == '__main__':
    render()
    print("render done")
