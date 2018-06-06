"""
Author: Gabriel Garza
Date:   05/27/2018
Description: Based on main sumilation from Reuben Ferrante. Runs the simulation using Policy Gradient model.
"""

from environments.rocketlander import RocketLander
from constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT, DEGTORAD, NOZZLE_ANGLE_LIMIT
import numpy as np
from policy_gradient import PolicyGradient
from time import gmtime, strftime

RENDER_ENV = False
EPISODES = 10
rewards = []
RENDER_REWARD_MIN = 500


if __name__ == "__main__":
    # Settings holds all the settings for the rocket lander environment.
    settings = {'Side Engines': True,
                'Clouds': True,
                'Vectorized Nozzle': True,
                'Starting Y-Pos Constant': 1,
                'Initial Force': 'random'}  # (6000, -10000)}

    env = RocketLander(settings)

    # Load checkpoint
    load_version = 5
    load_version = 5
    timestamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    load_path = "output/model1/{}/RocketLander.ckpt".format(load_version)
    save_path = "output/model1/{}/RocketLander.ckpt".format(timestamp)

    action_bounds = [1, 1, 15*DEGTORAD]

    print("env.observation_space.shape[0]", env.observation_space.shape[0])
    print("env.action_space", len(env.action_space))
    print("action_bounds", action_bounds)

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = len(env.action_space),
        learning_rate=0.001,
        reward_decay=0.99,
        load_path=load_path,
        save_path=save_path,
        epsilon_max=0.95,
        epsilon_greedy_increment=0.01,
        initial_epsilon = 0.8
    )

    observation = env.reset()

    left_or_right_barge_movement = np.random.randint(0, 2)
    epsilon = 0.05


    for episode in range(EPISODES):
        while True:
            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 3. Store transition for training
            # if reward > -0.20:
            PG.store_transition(observation, action, reward)

            if RENDER_ENV:
                # -------------------------------------
                # Optional render
                env.render()
                # Draw the target
                env.draw_marker(env.landing_coordinates[0], env.landing_coordinates[1])
                # Refresh render
                env.refresh(render=False)

            # When should the barge move? Water movement, dynamics etc can be simulated here.
            if observation[LEFT_GROUND_CONTACT] == 0 and observation[RIGHT_GROUND_CONTACT] == 0:
                env.move_barge_randomly(epsilon, left_or_right_barge_movement)
                # Random Force on rocket to simulate wind.
                env.apply_random_x_disturbance(epsilon=0.005, left_or_right=left_or_right_barge_movement)
                env.apply_random_y_disturbance(epsilon=0.005)


            observation = observation_

            if done:

                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("Episode: ", episode)
                print("Epsilon: ", PG.epsilon)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)
                print("==========================================")

                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn()

                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True

                observation = env.reset()
                break
