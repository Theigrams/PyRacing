import argparse

import gymnasium as gym
import numpy as np

# from pyvirtualdisplay import Display
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import gym_race

# Virtual display
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--train", action="store_true")
    args.add_argument("--timesteps", type=int, default=5 * 1e5)
    args = args.parse_args()
    train_model = args.train
    timesteps = args.timesteps

    env = gym.make("Pyrace-v0")

    if train_model:
        env.unwrapped.set_view(False)
        model = PPO(
            "MlpPolicy",
            env,
            # batch_size=64,
            verbose=1,
            device="cuda",
            tensorboard_log="./tensorboard/",
        )
        eval_callback = EvalCallback(env, eval_freq=max(timesteps // 20, 1), n_eval_episodes=5, log_path="./logs/")
        model.learn(total_timesteps=timesteps, callback=eval_callback)
        model.save("ppo")
    else:
        env.unwrapped.set_view(True)
        model = PPO.load("ppo", env=env, custom_objects={"env": env, "clip_range": 0.2, "lr_schedule": "linear"})
        obs, info = env.reset()
        episode_reward = 0
        for t in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
            env.render()
        print("Reward:", episode_reward)
        print("Steps:", t)
