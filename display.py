from collections import deque

import os
import shutil
import argparse

import torch

from utils_env import MyEnv
from utils_drl import Agent


def render(target: int = 0) -> None:
    '''
    render a movie of target model
    '''

    model_name: str = f"model_{target:03d}"
    model_path: str = f"./models/{model_name}"

    tmp_frames_dir = "tmp_frames"
    movie_dir = "movie"

    if os.path.exists(tmp_frames_dir):
        shutil.rmtree(tmp_frames_dir)
    os.mkdir(tmp_frames_dir)

    if not os.path.exists(movie_dir):
        os.mkdir(movie_dir)

    print("load model...")
    device = torch.device("cpu")
    env = MyEnv(device)
    agent = Agent(
        action_dim=env.get_action_dim(),
        device=device,
        gamma=0.99,
        seed=0,
        restore=model_path)

    print("evaluate model...")
    obs_queue = deque(maxlen=5)
    avg_reward, frames = env.evaluate(obs_queue, agent, render=True)
    print(f"Avg. Reward: {avg_reward:.1f}")

    print("generate frames...")
    for ind, frame in enumerate(frames):
        frame.save(os.path.join(
            tmp_frames_dir, f"{ind:06d}.png"), format="png")

    print("generate movie with ffmpeg...")
    input_files = f'./{tmp_frames_dir}/%06d.png'
    output_file = f'./{movie_dir}/movie-{target:03d}.mp4'

    if os.system(f'''ffmpeg -loglevel error -i {input_files} -pix_fmt yuv420p -y {output_file}''') != 0:
        print("ffmpeg error")
    else:
        print(f"render output {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='display target model')
    parser.add_argument('-target', action="store", default=0, type=int)
    args = parser.parse_args()

    print("render begin...")
    render(target=args.target)
    print("render done")
