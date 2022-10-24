import gym
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

import sys
sys.path.append("..")
from utils.NormalizeActionWrapper import NormalizeActionWrapper
from envs.cassie.cassie import CassieRefEnv

if __name__ == '__main__':
    env = CassieRefEnv(visual=True, dynamics_randomization=False)
    env = NormalizeActionWrapper(env)


    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            if self.n_calls % 51200 == 0:
                print("Saving new best model")
                self.model.save(f"../model_saved/ppo_cassie_{self.n_calls}")

            return True


    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=[dict(pi=[512, 512], vf=[512, 512])])
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, policy_kwargs=policy_kwargs,
                batch_size=128, tensorboard_log="../log/")
    # model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./log/")
    model.is_tb_set = False

    model.learn(total_timesteps=4e7, n_eval_episodes=10, callback=TensorboardCallback())
    model.save("ppo_m02l")
