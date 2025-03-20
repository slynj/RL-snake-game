from stable_baselines3 import PPO # Proximal Policy Optimization
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
from stable_baselines3.common.env_checker import check_env
import sys

from src.model.SnakeGame import SnakeGame
from env.env import MODEL_DIR_PATH, MODEL_PATH, LOG_PATH

# python -m src.model.train 


def train():
    # env check
    env = SnakeGame()
    check_env(env, warn=True)

    # log
    # log_dir = os.path.abspath("../../log")
    os.makedirs(LOG_PATH, exist_ok=True)

    # env
    env = SnakeGame()

    # wrap env with monitor
    env = Monitor(env, LOG_PATH)

    # cb fn -> periodically evaluate the model and save the best version
    eval_cb = EvalCallback(env, best_model_save_path=MODEL_DIR_PATH,
                        log_path=LOG_PATH,
                        eval_freq=5000,
                        deterministic=False,
                        render=False)

    # PPO hyperparam
    PPO_model_args = {
        "learning_rate" : 0.002,
        "gamma" : 0.99, # discount factor for further rewards [0, 1]
        "verbose" : 0, # 1 -> more info on training steps
        "seed" : 523,
        "ent_coef" : 0.2, # entropy coef -> encourage exploration
        "clip_range" : 0.2 # limits p of action difference
    }

    # Multi Input Policy since we have 1+ states as an 'input'
    model = PPO('MultiInputPolicy', env, **PPO_model_args)

    # model_path = os.path.abspath("../../model/model.zip")

    if os.path.exists(MODEL_PATH):
        print("Loading pretrained model...", flush=True)
        model.set_parameters(MODEL_PATH)
    
    model.learn(6000000, callback=eval_cb)
    # model.learn(160000, callback=eval_cb)
    # model.save(model_path)
    
    sys.stdout.flush() 

if __name__ == "__main__":
    train()