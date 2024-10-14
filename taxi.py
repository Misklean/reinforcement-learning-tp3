"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from moviepy.editor import ImageSequenceClip
import os

# Create directory for videos if it doesn't exist
video_dir = "./videos/"
os.makedirs(video_dir, exist_ok=True)

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

# You can edit these hyperparameters!
agent = QLearningAgent(
    learning_rate=0.7, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()
    
    frames = []  # List to store frames

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # BEGIN SOLUTION
        # Capture frame for video
        frames.append(env.render())  # Capture the current frame
        
        # Update the Q-values of the agent based on the action taken and the reward
        agent.update(s, a, r, next_s)

        # Accumulate the reward
        total_reward += r

        # Transition to the next state
        s = next_s

        if done:
            break
        # END SOLUTION

    return total_reward, frames

def save_video(frames, filename, fps=30):
    """ Save frames as a video using moviepy """
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename, fps=fps)

# Run training with QLearningAgent and record video
rewards = []
for i in range(1000):
    total_reward, frames = play_and_train(env, agent)
    rewards.append(total_reward)
    
    if i % 100 == 0:
        print(f"mean reward: {np.mean(rewards[-100:])}")
        
        # Save video every 100 episodes
        video_path = os.path.join(video_dir, f"taxi_qlearning_episode_{i}.mp4")
        save_video(frames, video_path, fps=30)

assert np.mean(rewards[-100:]) > 0.0

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    total_reward, frames = play_and_train(env, agent)
    rewards.append(total_reward)
    
    if i % 100 == 0:
        print(f"mean reward: {np.mean(rewards[-100:])}")
        
        # Save video every 100 episodes
        video_path = os.path.join(video_dir, f"taxi_qlearning_eps_episode_{i}.mp4")
        save_video(frames, video_path, fps=30)

assert np.mean(rewards[-100:]) > 0.0


####################
# 3. Play with SARSA
#################### 

agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    total_reward, frames = play_and_train(env, agent)
    rewards.append(total_reward)
    
    if i % 100 == 0:
        print(f"mean reward: {np.mean(rewards[-100:])}")
        
        # Save video every 100 episodes
        video_path = os.path.join(video_dir, f"taxi_sarsa_episode_{i}.mp4")
        save_video(frames, video_path, fps=30)

env.close()
