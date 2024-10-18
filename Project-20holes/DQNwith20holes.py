import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()


        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class FrozenLakeDQL():

    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32


    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # FrozeLake environment training
    def train(self, episodes, render=False, is_slippery=False):
        # FrozenLake instance with custom 10x10 map
        custom_map = self.create_custom_map(size=10, num_holes=20)
        env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=is_slippery, render_mode='human' if render else None)

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)


        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)


        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy before training):')
        self.print_dqn(policy_dqn)


        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)


        rewards_per_episode = np.zeros(episodes)


        epsilon_history = []


        step_count=0

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False


            while(not terminated and not truncated):

                # Select random action
                if random.random() < epsilon:

                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()


                new_state,reward,terminated,truncated,_ = env.step(action)


                memory.append((state, action, new_state, reward, terminated))

                state = new_state


                step_count+=1

            #  rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)


                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)


                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0


        env.close()

  #Saving with [LAST_NAME]-model.pt , file.
        torch.save(policy_dqn.state_dict(), "Chhetri-model.pt")


        plt.figure(1)


        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121)
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)
        plt.plot(epsilon_history)

        plt.savefig('frozen_lake_dql.png')


    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # FrozeLake environment
    def test(self, episodes, is_slippery=False):
        custom_map = self.create_custom_map(size=10, num_holes=20)
        env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval()

        print('Trained policy:')
        self.print_dqn(policy_dqn)
        success_count= 0
        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            trajectory =[]

            while(not terminated and not truncated):
                trajectory.append((state // 10, state % 10))
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state,num_states)).argmax().item()

                state,reward,terminated,truncated,_ = env.step(action)
            if reward == 1:
               success_count += 1
               print(f"Episode {i+1}: {'Success'} , Trajectory: {trajectory}")

        env.close()

        print(f'   QUES.3    Success rate: {(success_count/episodes)*100}%')

    def print_dqn(self, dqn):
        num_states = dqn.fc1.in_features
        for s in range(num_states):
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s,num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '
            q_values=q_values.rstrip()

            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]
            # Calculate grid coordinates (x, y)
            x = s // 10
            y = s % 10
            print(f'({x},{y}),action:{best_action} |', end='')
            '''print(f'{s:02}:Best action:{best_action}|',end='')'''
            if (s+1)%10==0:
                print() #newline after 4

    #Creating a custom map with 20 holes and 10*10 matrix size
    def create_custom_map(self, size=10, num_holes=20):

        custom_map = [['F' for _ in range(size)] for _ in range(size)]

        # Pather id:002831347 STarting point 1,3 and ending point 4,7
        custom_map[1][3] = 'S'
        custom_map[4][7] = 'G'
        # Randomly place holes
        holes = set()
        while len(holes) < num_holes:
            x = random.randint(0, size-1)
            y = random.randint(0, size-1)

            if (x, y) not in holes and (x, y) not in [(1, 3), (4, 7)]:
                custom_map[x][y] = 'H'  # Place a hole
                holes.add((x, y))

        return ["".join(row) for row in custom_map]

# Giving the number of test episodes to be 40 as described in project requirement
if __name__ == '__main__':

    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    frozen_lake.train(1000, is_slippery=is_slippery)
    is_slippery = True
    frozen_lake.test(40, is_slippery=is_slippery)

