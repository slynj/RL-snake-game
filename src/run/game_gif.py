
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

# from src.model.train import env, model
from src.model.SnakeGame import SnakeGame


# python -m src.run.game_gif 


env = SnakeGame()
env = Monitor(env, "../../log")

model_path = "../../best_model/best_model.zip"
if os.path.exists(model_path):
    print("Loading model...")
    model = PPO.load(model_path, env=env)
else:
    raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")


obs, _ = env.reset()

# for img gif
fig, ax = plt.subplots(figsize=(6, 6))
plt.axis('off')
frames = []
fps = 18
# plt.rcParams['animation.writer'] = 'ffmpeg'

n_steps = 1000000
total_reward = 0

for step in range(n_steps):
    # preprocess the obs to match the model's input format
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, trunc, info = env.step(action)
    
    total_reward += reward

    print(f"Step {step + 1} \nAction: {action} \nTotal Reward: {total_reward}")
    
    frames.append([ax.imshow(env.unwrapped.render(mode='rgb_array'), animated=True)])

    if done:
        print(f"Game Over! Total Reward: {total_reward}")
        break

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

anim = animation.ArtistAnimation(fig, frames, interval=int(1000/fps), blit=True, repeat_delay=1000)
anim.save("snake_game.gif", dpi=150)
