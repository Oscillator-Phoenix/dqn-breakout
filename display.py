from collections import deque

import os
import shutil
import argparse

import torch

from utils_env import MyEnv
from utils_drl import Agent


# 设置模型持久化的路径
parser = argparse.ArgumentParser(description='load target model')
parser.add_argument('-target', action="store", default=0, type=int)
parser.add_argument('-dueling', action="store", default=False, type=bool)

args = parser.parse_args()
args_target: int = args.target
args_dueling: bool = args.dueling

SAVE_PREFIX = "./dueling_dqn__models" if args_dueling else "./models"

if not os.path.exists(SAVE_PREFIX):
    os.mkdir(SAVE_PREFIX)
model_name: str = f"model_{args_target:03d}"
model_path: str = os.path.join(SAVE_PREFIX, model_name)
print(f"load target model: {model_path}")

tmp_frames_dir = "./tmp_frames"
movie_dir = "./dueling_dqn_movie" if args_dueling else "./movie"

if os.path.exists(tmp_frames_dir):
    shutil.rmtree(tmp_frames_dir)
os.mkdir(tmp_frames_dir)

if not os.path.exists(movie_dir):
    os.mkdir(movie_dir)


def render() -> None:
    '''
    render a movie of target model
    '''

    print("load model...")
    device = torch.device("cpu")
    env = MyEnv(device)
    agent = Agent(
        action_dim=env.get_action_dim(),
        device=device,
        gamma=0.99,
        seed=0,
        restore=model_path,
        q_func="DuelingDQN" if args_dueling else None
    )

    print("evaluate model...")
    obs_queue = deque(maxlen=5)
    avg_reward, frames = env.evaluate(obs_queue, agent, render=True)
    print(f"Avg. Reward: {avg_reward:.1f}")

    print("generate frames...")
    for ind, frame in enumerate(frames):
        frame.save(os.path.join(
            tmp_frames_dir, f"{ind:06d}.png"), format="png")

    print("generate movie with ffmpeg...")
    input_files = f'{tmp_frames_dir}/%06d.png'
    output_file = f'{movie_dir}/movie-{args_target:03d}.mp4'

    if os.system(f'''ffmpeg -loglevel error -i {input_files} -pix_fmt yuv420p -y {output_file}''') != 0:
        print("ffmpeg error")
    else:
        print(f"render output {output_file}")


if __name__ == '__main__':

    print("render begin...")
    render()
    print("render done")
