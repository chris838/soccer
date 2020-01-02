import pdb
import pickle
import torch
import random

import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter

from replay_buffer import ReplayBuffer
from maddpg_agent import MaddpgAgent


class Trainer():

    def __init__(self, env, replay_buffer,
            discount = 0.99,
            tau = 0.01,
            actor_lr = 1e-4,
            critic_lr = 3e-4):

        self.env = env
        self.replay_buffer = replay_buffer

        # Parse the environment info
        self.brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        global_state_space_size = \
            env_info.vector_observations.flatten().shape[0]
        global_action_space_size = \
            env_info.previous_vector_actions.flatten().shape[0]
        brain = env.brains[self.brain_name]
        action_space_size = brain.vector_action_space_size
        state_space_size = \
            brain.num_stacked_vector_observations * brain.vector_observation_space_size

        # Create the agents
        self.agents = []
        for i in range(self.num_agents):
            print(f"Agent {i}: state space: {state_space_size}; \
                    action space {action_space_size}.")
            self.agents.append(MaddpgAgent(
                i, self.num_agents, state_space_size, action_space_size,
                global_state_space_size, global_action_space_size,
                discount=discount, tau=tau, actor_lr=actor_lr,
                critic_lr=critic_lr,
                ))

        # Track progres
        self.episode_returns = []
        self.loss = []
        self.writer = SummaryWriter()

        # Training vars
        self.train_step = 0
        self.episode = 0
        self.is_learning = False


    def train(self,
              num_episodes=1500,
              batch_size=1024,
              max_episode_length=250,
              train_every_steps=100,
              noise_level=2.0,
              noise_decay=0.9999,
              print_episodes=100
              ):

        try:

            print(f"------------------------------------------------")
            print(f"New training run.")
            print(f"    num_episodes: {num_episodes}")
            print(f"    batch_size: {batch_size}")
            print(f"    max_episode_length: {max_episode_length}")
            print(f"    replay_buffer_size: {len(self.replay_buffer)}")
            print(f"    train_every_steps: {train_every_steps}")
            print(f"    noise_level: {noise_level}")
            print(f"    noise_decay: {noise_decay}")

            # Iterate over episodes
            episode_max = self.episode + num_episodes
            while self.episode < episode_max:

                # Receive initial state vector s
                #   s = (s_1, . . . , s_N)
                env_info = self.env.reset(train_mode=True)[self.brain_name]
                s = env_info.vector_observations

                self.episode_returns.append(np.array([0] * self.num_agents))
                for t in range(1, max_episode_length):

                    # For each agent i, select actions:
                    #   a = (a_1, . . . , a_N)
                    # using the current policy and exploration noise, which we decay
                    a = [agent.act(state, noise_level=noise_level)
                         for agent, state in zip(self.agents, s)]
                    if self.is_learning:
                        noise_level *= noise_decay

                    # Execute actions a = (a_1, . . . , a_N)
                    # Observe:
                    #   Reward r = (r_1, . . . , r_N)
                    #   Next-state vector s' = (s'_1, . . . , s'_N)
                    env_info = self.env.step(a)[self.brain_name]
                    r = env_info.rewards
                    s_prime = env_info.vector_observations
                    dones = env_info.local_done

                    # Store (s, a, r, s', done) in replay buffer
                    self.replay_buffer.append((s, a, r, s_prime, dones))

                    # Record progress
                    self.episode_returns[-1] = self.episode_returns[-1] + r

                    # Advance
                    s = s_prime
                    self.train_step += 1

                    # Periodically (after a certain number of steps) run update/training
                    if self.train_step % train_every_steps == 0:
                        if self.replay_buffer.has_enough_samples():

                            if not self.is_learning:
                                print(f"Started learning at time {self.train_step}")
                                self.is_learning = True

                            # Sample replay buffer
                            sample = self.replay_buffer.sample(
                                batch_size=batch_size)

                            # For every sample tuple, each agent needs to know which action
                            # would be chosen under the policy of the other agents in the
                            # next state s', in order to calculate q-values.
                            next_actions = [[
                                agent.act(next_state, target_actor=True)
                                for agent, next_state in zip(self.agents, s_prime)]
                                for (s, a, r, s_prime, dones) in sample]

                            # Update/train all the agents
                            per_agent_loss = []
                            for agent in self.agents:
                                actor_loss, critic_loss = agent.update(
                                    sample, next_actions)
                                per_agent_loss.append(
                                    (actor_loss, critic_loss))
                            self.loss.append(per_agent_loss)

                    # Terminate episode early if done
                    if any(dones):
                        break

                self.episode += 1
                if self.episode % print_episodes == 0:
                    self.log_parameters()
                    print(f"t: {self.train_step}, e: {self.episode}, noise: {noise_level:.2f}. " +
                          f"Average last {print_episodes} episode return: " +
                          f"{np.array(self.episode_returns[-print_episodes:]).mean(axis=0)}")

            print("Finished")

        except KeyboardInterrupt:
            print("Interrupted")

    def log_parameters(self):
        for agent in self.agents:
            for network_name, network in [('actor', agent.actor), ('critic', agent.critic)]:
                for name, param in network.named_parameters():
                    label = 'agent_' + str(agent.i) + '/' + network_name + '/' + name
                    self.writer.add_histogram(label, param.clone().cpu().data.numpy(), self.train_step)

    def get_average_loss(self):
        if len(self.loss) > 0:
            return np.array(self.loss).mean(axis=1)
        return [[0, 0]]

    def get_max_returns(self):
        if len(self.episode_returns) > 0:
            return np.array(self.episode_returns).max(axis=1)
        return []
