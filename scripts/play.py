import gym
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from omegaconf import DictConfig, OmegaConf
import hydra


import sys
sys.path.append("..")
from utils.NormalizeActionWrapper import NormalizeActionWrapper
from envs.cassie.cassie import CassieRefEnv



@hydra.main(version_base=None, config_path="../envs/cassie", config_name="config")
def run_play(cfg: DictConfig) -> None:
    env = CassieRefEnv(cfg=cfg)
    env = NormalizeActionWrapper(env)
    model = PPO.load("../log/cassie/2022-07-28-21-03-23/model_saved/ppo_cassie_2099200.zip", env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)


if __name__ == '__main__':
    run_play()