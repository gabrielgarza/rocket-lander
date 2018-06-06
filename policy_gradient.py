"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from constants import NOZZLE_ANGLE_LIMIT


class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None,
        epsilon_max=0.95,
        epsilon_greedy_increment=0.001,
        initial_epsilon = 0.8
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max
        self.epsilon_greedy_increment = epsilon_greedy_increment

        if epsilon_greedy_increment is not None:
            self.epsilon = initial_epsilon
        else:
            self.epsilon = self.epsilon_max

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()

        self.cost_history = []

        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        if np.random.uniform() < self.epsilon:
            # print("Predict Action")

            # Reshape observation to (num_features, 1)
            observation = observation[:, np.newaxis]

            # Run forward propagation to get softmax probabilities
            actions = self.sess.run(self.outputs, feed_dict = {self.X: observation})
        else:
            # print("Take Random Action")
            actions = np.array([np.random.uniform(0, 1),np.random.uniform(-1, 1),np.random.uniform(-NOZZLE_ANGLE_LIMIT, NOZZLE_ANGLE_LIMIT)])

        return actions.ravel()

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: np.vstack(self.episode_observations).T,
             self.Y: np.vstack(np.array(self.episode_actions)).T,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            print("Model saved in file: %s" % save_path)

        # Increase epsilon to make it more likely over time to get actions from predictions instead of from random sample
        if self.epsilon_greedy_increment is not None:
            self.epsilon = np.round(min(self.epsilon_max, self.epsilon + self.epsilon_greedy_increment),3)

        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0
        # for t in reversed(range(len(self.episode_rewards))):
        #     cumulative = cumulative * self.gamma + self.episode_rewards[t]
        #     discounted_episode_rewards[t] = cumulative

        discounted_episode_rewards -= np.mean(self.episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return self.episode_rewards


    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        units_layer_1 = 10
        units_layer_2 = 10
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1,self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)

        self.outputs = logits

        with tf.name_scope('loss'):
            neg_log_prob = tf.losses.mean_squared_error(labels, logits)
            loss = neg_log_prob * self.discounted_episode_rewards_norm  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
