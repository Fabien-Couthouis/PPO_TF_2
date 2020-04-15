# by F. Couthouis
import numpy as np
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, CategoricalCrossentropy


NN_HIDDEN_SIZE = 64  # hidden size of neurons layers
LR = 1e-4  # learning rate
EPS_CLIP = 0.2  # clipping value
CRITIC_FACTOR = 0.5  # factor applied to critic loss in total loss computation
ENTROPY_B = 1e-3  # entropy factor in loss calculation
GAMMA = 0.99  # discount factor


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []

    def forget(self):
        'Clear memory'
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logits.clear()

    def remember(self, state, action, reward, done, logits):
        'Apend lists into memory'
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logits.append(logits)

    def get_states_rewards_dones(self):
        '! states are returned as np array'
        return np.array(self.states), self.rewards, self.dones

    def get_states_logits_actions_rewards_tensors(self):
        return (tf.convert_to_tensor(self.states, dtype=tf.float32),
                tf.convert_to_tensor(self.logits, dtype=tf.float32),
                tf.convert_to_tensor(self.actions, dtype=tf.float32),
                tf.convert_to_tensor(self.rewards, dtype=tf.float32))


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions, hidden_size):
        super().__init__('mlp_policy')

        # Actor
        self.hidden11 = Dense(hidden_size, activation='relu', name='h11')
        self.hidden12 = Dense(hidden_size, activation='relu', name='h12')
        self.action = Dense(num_actions, name="action", activation="softmax")

        # Critic
        self.hidden21 = Dense(hidden_size, activation='relu', name='h21')
        self.hidden22 = Dense(hidden_size, activation='relu', name='h22')
        self.value = Dense(1, name='value', activation="tanh")

    def call(self, x):
        # Actor
        hidden_logs = self.hidden11(x)
        hidden_logs = self.hidden12(hidden_logs)

        # Critic
        hidden_vals = self.hidden21(x)
        hidden_vals = self.hidden22(hidden_vals)

        return self.action(hidden_logs), self.value(hidden_vals)

    def get_values(self, states):
        _, values = self.predict(states)
        return values

    def actions_logits_values(self, states):
        action_probs, values = self.predict(states)
        dist = tfp.distributions.Categorical(probs=action_probs)
        actions = dist.sample()
        # logprobs/logits are normalized log probabilities
        action_logprobs = dist.log_prob(actions)
        return np.squeeze(actions), np.squeeze(action_logprobs), np.squeeze(values)

    def evaluate(self, states, actions):
        """
        Evaluate states (actor) and actions (critic)
        return: actions_logprobs, values, dist_entropy
        """
        action_probs, values = self(states)
        dist = tfp.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return action_logprobs, values, dist_entropy


class PPO_Agent():
    def __init__(self, num_actions):
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.critic_factor = CRITIC_FACTOR
        self.entropy_b = ENTROPY_B
        self.memory = Memory()
        self.model = ActorCritic(num_actions, NN_HIDDEN_SIZE)
        self.optimizer = Adam(lr=LR)

    def train_step(self, nb_epochs=16):
        states, oldpolicy_probs, actions, rewards = self.memory.get_states_logits_actions_rewards_tensors()
        advantages = self.get_advantages()

        for _ in range(nb_epochs):
            with tf.GradientTape() as tape:
                newpolicy_probs, values, dist_entropy = self.model.evaluate(
                    states, actions)

                # Compute total PPO loss
                actor_loss = self.get_actor_loss(
                    newpolicy_probs, oldpolicy_probs, advantages)

                critic_loss = tf.keras.losses.mean_squared_error(
                    values, rewards)

                total_loss = self.critic_factor * critic_loss + \
                    actor_loss - self.entropy_b * dist_entropy

            gradients = tape.gradient(
                total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

    def get_actor_loss(self, newpolicy_probs, oldpolicy_probs, advantages):
        'PPO loss (actor part)'
        ratio = tf.exp(newpolicy_probs - oldpolicy_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1-self.eps_clip, 1+self.eps_clip)
        surr2 = surr2 * advantages
        return -tf.math.reduce_mean(tf.math.minimum(surr1, surr2))

    def get_advantages(self):
        states, _, _ = self.memory.get_states_rewards_dones()
        _, _, est_values = self.model.actions_logits_values(states)
        true_values = self.get_returns()

        # Compute advantage array (this is also the TD error)
        advantages = true_values - est_values
        return (advantages - advantages.mean()) / (advantages.std()+1e-8)

    def choose_action(self, observation):
        'return: action, logprobs'
        action, logprobs, _ = self.model.actions_logits_values(
            np.expand_dims(observation, axis=0))
        return action, logprobs

    def get_returns(self):
        states, rewards, dones = self.memory.get_states_rewards_dones()

        # If last state not terminal, estimate v(s) using the critic
        last_state = np.expand_dims(states[-1], axis=0)
        returns = np.zeros(len(rewards)+1)
        returns[-1] = self.model.get_values(last_state) if not dones[-1] else 0

        # returns = discounted sum of future rewards
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * \
                returns[t+1] if not dones[t] else 0

        returns = returns[:-1]
        return (returns - returns.mean()) / (returns.std() + 1e-8)


class Trainer:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, nb_episodes=1000, nb_steps=128):
        all_rewards, current_rewards = [], []
        finished_games = 0
        observation = self.env.reset()

        while finished_games <= nb_episodes:
            for _step in range(nb_steps):
                # self.env.render()
                action, logits = self.agent.choose_action(observation)

                next_observation, reward, done, _ = self.env.step(action)
                self.agent.memory.remember(
                    observation, action, reward, done, logits)
                current_rewards.append(reward)

                if done:
                    observation = self.env.reset()
                    finished_games += 1
                    all_rewards.append(sum(current_rewards))

                    if finished_games % 10 == 0:
                        print("==========================================")
                        print(
                            f"Game n°{finished_games} finished")
                        print("Reward: ", sum(current_rewards))
                        print("Mean Reward", sum(
                            all_rewards) / len(all_rewards))
                        print("Max reward so far: ", max(all_rewards))

                    current_rewards.clear()

                else:
                    observation = next_observation
            # train models
            self.algo.train_step()
            self.algo.memory.forget()

        self.env.close()


if __name__ == "__main__":
    agent = PPO_Agent(num_actions=2)  # 0: left / 1: right
    env = gym.make('CartPole-v1')
    trainer = Trainer(env, agent)
    trainer.train()
