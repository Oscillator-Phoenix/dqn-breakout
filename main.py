from collections import deque
import os
import random
import argparse

from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99

GLOBAL_SEED = 0

RENDER = False


STACK_SIZE = 4
MEM_SIZE = 100_000


BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000

# BATCH_SIZE = 32
# POLICY_UPDATE = 4
# TARGET_UPDATE = 10_000
# WARM_STEPS = 50_000
# MAX_STEPS = 10_000
# EVALUATE_FREQ = 1_000


# 初始化随机数
rand = random.Random()
rand.seed(GLOBAL_SEED)


def new_seed():
    return rand.randint(0, 1000_000)


torch.manual_seed(new_seed())


# 设置训练设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"available cpu threads: {torch.get_num_threads()}")
torch.set_num_threads(2)
print(f"use cpu threads: {torch.get_num_threads()}")


# 设置模型持久化的路径
SAVE_PREFIX = "./models"
parser = argparse.ArgumentParser(description='load target model')
parser.add_argument('-target', action="store", default=0, type=int)
args = parser.parse_args()
target: int = args.target
if not os.path.exists(SAVE_PREFIX):
    os.mkdir(SAVE_PREFIX)
model_name: str = f"model_{target:03d}"
model_path: str = os.path.join(SAVE_PREFIX, model_name)
print(f"load target model: {model_path}")


# Env, Agent
env = MyEnv(device)
agent = Agent(
    action_dim=env.get_action_dim(),
    device=device,
    gamma=GAMMA,
    seed=new_seed(),
    restore=model_path
)


def epsilon_greedy_with_decayed() -> None:
    '''
    epsilon-greedy with epsilon decayed
    '''

    EPS_START: float = 1.0
    EPS_END: float = 0.1
    EPS_DECAY: float = 1000000

    # 初始化 epsilon 参数
    epsilon: float = EPS_START  # explorate rate
    epsilon_step: float = (EPS_START - EPS_END) / EPS_DECAY
    random_x = random.Random()  # x to test explorate rate
    random_x.seed(new_seed())

    # 初始化训练数据
    memory = ReplayMemory(
        channels=STACK_SIZE + 1,
        capacity=MEM_SIZE,
        device=device)
    obs_queue: deque = deque(maxlen=5)
    done = True

    # TODO warm up

    # 迭代
    for step in tqdm(range(MAX_STEPS), total=MAX_STEPS, ncols=50, leave=False, unit="b"):

        # if done, 重置 Env
        if done:
            observations, _, _ = env.reset()
            for obs in observations:
                obs_queue.append(obs)

        # agent 与 env 交互产生训练数据
        # 从 Env 获取一个 state
        state = env.make_state(obs_queue).to(device).float()
        # 根据 epsilon-greedy with epsilon decayed 方法产生训练数据
        action = agent.run_boltzmann(state)

        # action = agent.run_greedy(state, epsilon=epsilon)
        # epsilon = max(epsilon - epsilon_step, EPS_END)  # epsilon update

        obs, reward, done = env.step(action)
        obs_queue.append(obs)
        # 添加 数据 到 Replay 数据库
        memory.push(env.make_folded_state(obs_queue), action, reward, done)

        # 从训练数据库中抽取数据进行训练
        if step % POLICY_UPDATE == 0:
            agent.learn(memory, BATCH_SIZE)  # TODO priority

        # 权重同步
        if step % TARGET_UPDATE == 0:
            agent.sync()

        if step % EVALUATE_FREQ == 0:
            avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
            with open("rewards.txt", "a") as fp:
                fp.write(
                    f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
            if RENDER:
                prefix = f"eval_{step//EVALUATE_FREQ:03d}"
                os.mkdir(prefix)
                for ind, frame in enumerate(frames):
                    with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                        frame.save(fp, format="png")
            agent.save(os.path.join(
                SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
            done = True


def boltzmann_exploration() -> None:

    # boltzmann_exploration 参数
    _lambda: float = 1.0

    # 训练参数
    warm_up_step: int = 1_000
    train_step: int = 5_000_000
    sample_times_per_step: int = 8
    train_times_per_step: int = 2
    steps_for_target_update: int = 1_000
    steps_for_evaluate: int = 5_000
    memory_size: int = 50_000

    # 初始化训练数据
    memory = ReplayMemory(
        channels=STACK_SIZE + 1,
        capacity=memory_size,
        device=device)
    obs_queue: deque = deque(maxlen=5)
    done: bool = True

    # 采样函数
    def sample(done: bool):
        # if done, 重置 Env
        if done:
            observations, _, _ = env.reset()
            for obs in observations:
                obs_queue.append(obs)

        # agent 与 env 交互产生训练数据
        # 从 Env 获取一个 state
        state = env.make_state(obs_queue).to(device).float()
        action = agent.run_boltzmann(state, _lambda=_lambda)
        obs, reward, done = env.step(action)
        obs_queue.append(obs)
        # 添加 数据 到 Replay 数据库
        memory.push(env.make_folded_state(obs_queue), action, reward, done)

    # warm up
    print("warm up...")
    for step in tqdm(range(warm_up_step), total=warm_up_step, ncols=50, leave=False, unit="sample"):
        # sample(done=done)
        pass

    # 迭代
    print("tarin...")
    for step in tqdm(range(train_step), total=train_step, ncols=50, leave=False, unit="step"):

        # 采样
        for _ in range(sample_times_per_step):
            sample(done=done)

        # 训练
        for _ in range(train_times_per_step):
            # 从训练数据库中抽取数据进行训练
            agent.learn(memory, BATCH_SIZE)  # TODO priority

        # 权重同步
        if step % steps_for_target_update == 0:
            agent.sync()

        # 模型持久化
        if step % steps_for_evaluate == 0:
            avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
            with open("rewards.txt", "a") as fp:
                fp.write(
                    f"{step//steps_for_evaluate:3d} {step:8d} {avg_reward:.1f}\n")

            if RENDER:
                prefix = f"eval_{step//steps_for_evaluate:03d}"
                os.mkdir(prefix)
                for ind, frame in enumerate(frames):
                    with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                        frame.save(fp, format="png")

            agent.save(os.path.join(
                SAVE_PREFIX, f"model_{step//steps_for_evaluate:03d}"))

            done = True


if __name__ == '__main__':
    # print("epsilon_greedy_with_decayed")
    # epsilon_greedy_with_decayed()

    print("boltzmann_exploration")
    boltzmann_exploration()
