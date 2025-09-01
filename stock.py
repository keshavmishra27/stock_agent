from operator import index
import random
import pandas as pd
import numpy as np
import torch
from collections import deque
import copy
from IPython.display import clear_output


data = pd.read_csv(r"YNDX_150101_151231.csv")

columns = {
    "<DATE>": "DATE",
    "<TIME>": "TIME",
    "<OPEN>": "OPEN",
    "<HIGH>": "HIGH",
    "<CLOSE>": "CLOSE",
    "<LOW>": "LOW",
    "<VOL>": "VOL"
}
data_new = data.rename(columns=columns)

del data_new["DATE"], data_new["TIME"], data_new["VOL"]

data_new['remove'] = data_new.apply(lambda row: all([abs(i - row[0]) < 1e-8 for i in row]), axis=1)
data2 = data_new.query("remove == False").reset_index(drop=True)
del data_new["remove"]

data2["HIGH"] = (data2['HIGH'] - data2["OPEN"]) / data2["OPEN"]
data2["LOW"] = (data2['LOW'] - data2["OPEN"]) / data2["OPEN"]
data2["CLOSE"] = (data2['CLOSE'] - data2["OPEN"]) / data2["OPEN"]


class Yand():
    def __init__(self, data, obs_bars=10, test=False, commission_perc=0.3):
        self.data = data
        self.obs_bars = obs_bars
        self.have_pos = 0
        self.test = test
        self.commission_perc = commission_perc

        if not test:
            self.curr_step = np.random.choice(self.data.HIGH.shape[0] - self.obs_bars * 10) + self.obs_bars
        else:
            self.curr_step = self.obs_bars

        self.state = self.data[self.curr_step - self.obs_bars: self.curr_step]

    def step(self, action):
        reward = 0
        done = False
        relative_close = self.state["CLOSE"].iloc[-1]
        open_price = self.state["OPEN"].iloc[-1]
        close = open_price * (1 + relative_close)

        if action == "buy" and self.have_pos == 0:
            self.have_pos = 1
            self.open_price = close
            reward = -self.commission_perc

        elif action == "close" and self.have_pos == 1:
            reward = -self.commission_perc
            if not self.test:
                done = True
            reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_pos = 0
            self.open_price = 0.0

        self.curr_step += 1
        self.state = self.data[self.curr_step - self.obs_bars: self.curr_step]

        if self.curr_step == len(self.data):
            done = True

        state = np.zeros((5, self.obs_bars), dtype=np.float32)
        state[0] = self.state.HIGH.to_list()
        state[1] = self.state.LOW.to_list()
        state[2] = self.state.CLOSE.to_list()
        state[3] = self.have_pos
        if self.have_pos:
            state[4] = (close - self.open_price) / self.open_price

        return state, reward, done


actions = {0: "do_nothing", 1: "buy", 2: "close"}

Yand_env = Yand(data2, test=False, obs_bars=30)
state, _, _ = Yand_env.step("do_nothing")
print(state.shape)
input_shape = state.shape  # (channels, obs_bars)


def update(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def pre_process_state(state, in_shape, add_noise=False):
    if add_noise:
        state = state.reshape(in_shape) + np.random.rand(*in_shape) / 10.0
    else:
        state = state.reshape(*in_shape)
    if len(state.shape) == 2:
        state = state[np.newaxis, ...]
    return state


def get_action(q_val, num_act, epsilon):
    if random.random() < epsilon:
        action = np.random.randint(0, num_act)
    else:
        action = np.argmax(q_val, 1)[0]
    return action


class ReplayBuffer:
    def __init__(self, capacity, n_steps=3, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma

    def add(self, transition):
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        indices = []
        while len(indices) < batch_size:
            idx = random.randint(0, len(self.buffer) - self.n_steps - 1)
            done_in_trajectory = any(self.buffer[idx + i][4] for i in range(self.n_steps))
            if not done_in_trajectory:
                indices.append(idx)

        state1_batch, action_batch, reward_batch, state2_batch, done_batch = [], [], [], [], []
        for idx in indices:
            s, a, _, _, _ = self.buffer[idx]
            cum_reward = 0.0
            for i in range(self.n_steps):
                r = self.buffer[idx + i][2]
                cum_reward += (self.gamma ** i) * r
            s_n, _, _, _, d_n = self.buffer[idx + self.n_steps]
            state1_batch.append(s)
            action_batch.append(a)
            reward_batch.append(cum_reward)
            state2_batch.append(s_n)
            done_batch.append(d_n)

        state1_batch = torch.cat(state1_batch)
        action_batch = torch.tensor(action_batch, dtype=torch.int64)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        state2_batch = torch.cat(state2_batch)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        return state1_batch, action_batch, reward_batch, state2_batch, done_batch


def batch_target_for_nsteps_dqn(reward_batch, gamma, maxQ, n_steps_done_batch):
    Y = reward_batch + (gamma * maxQ * (1 - n_steps_done_batch))
    return Y


class deuling_conv1d_Q_net(torch.nn.Module):
    def __init__(self, in_depth_len, out_shp, obs_bars):
        super(deuling_conv1d_Q_net, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_depth_len, 128, 5)
        self.conv2 = torch.nn.Conv1d(128, 128, 5)
        self.activation = torch.nn.ReLU()
        self.flattern = torch.nn.Flatten()

        with torch.no_grad():
            x = torch.zeros(1, in_depth_len, obs_bars)
            x = self.conv1(x)
            x = self.activation(x)
            x = self.conv2(x)
            x = self.activation(x)
            x = self.flattern(x)
            conv_out_size = x.shape[1]

        self.state_val_linear1 = torch.nn.Linear(conv_out_size, 512)
        self.state_val_linear2 = torch.nn.Linear(512, 1)
        self.advantage_linear1 = torch.nn.Linear(conv_out_size, 512)
        self.advantage_linear2 = torch.nn.Linear(512, out_shp)

    def forward(self, s):
        s = self.conv1(s)
        s = self.activation(s)
        s = self.conv2(s)
        s = self.activation(s)
        s = self.flattern(s)
        s_state_val = self.state_val_linear1(s)
        s_state_val = self.activation(s_state_val)
        s_state_val = self.state_val_linear2(s_state_val)
        s_advantage = self.advantage_linear1(s)
        s_advantage = self.activation(s_advantage)
        s_advantage = self.advantage_linear2(s_advantage)
        y = s_state_val + s_advantage - s_advantage.mean(dim=1, keepdim=True)
        return y


# Initialize components
memory_size = 100
batch_size = 30
gamma = 0.99
lr = 0.0001
sync_freq = 10
output_shape = len(actions)
epsilon = 1.0
k = 0

replay = ReplayBuffer(capacity=memory_size, n_steps=3, gamma=gamma)

agentNN = deuling_conv1d_Q_net(input_shape[0], output_shape, obs_bars=input_shape[1])
targetNN = copy.deepcopy(agentNN)
targetNN.load_state_dict(agentNN.state_dict())

optimizer = torch.optim.RMSprop(agentNN.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

while k < 300:
    game = Yand(data2, commission_perc=0.1, obs_bars=30)
    state1, _, _ = game.step("do_nothing")
    state1 = np.array(state1)
    state1 = pre_process_state(state1, (1 * input_shape), add_noise=False)
    state1 = torch.from_numpy(state1).float()
    status = 1
    episodes_rewards = []
    while status == 1:
        k += 1
        with torch.no_grad():
            qval = agentNN(state1)
            qval_ = qval.numpy()

        action = get_action(qval_, 3, epsilon)
        action_name = actions[action]
        state2, reward, done = game.step(action_name)
        state2 = np.array(state2)
        state2 = pre_process_state(state2, (1 * input_shape), add_noise=False)
        state2 = torch.from_numpy(state2).float()
        points = (state1, action, reward, state2, done)
        episodes_rewards.append(reward)
        replay.add(points)

        if len(replay) >= batch_size:
            state1_batch, action1_batch, reward_batch, state2_batch, done_batch = replay.sample(batch_size)

            state_batch = torch.cat([state1_batch, state2_batch], dim=0)
            Q_1_and_n = agentNN(state_batch)
            Q1 = Q_1_and_n[:batch_size]

            with torch.no_grad():
                Q_nest_state = Q_1_and_n[batch_size:]
                selected_nodes_for_target_network = Q_nest_state.max(1)[1]
                Q_nest_state = targetNN(state2_batch)

            best_Q_nest_state = Q_nest_state.gather(dim=1, index=selected_nodes_for_target_network.unsqueeze(1)).squeeze()
            y = batch_target_for_nsteps_dqn(reward_batch, gamma, best_Q_nest_state, done_batch)
            x = Q1.gather(dim=1, index=action1_batch.unsqueeze(1)).squeeze()
            loss = loss_fn(x, y)
            print(f"step: {k}, mean reward: {np.mean(episodes_rewards):.2f}, loss: {loss.item():.4f}")

            clear_output(wait=True)
            update(loss, optimizer)

            state1 = state2

            if k % sync_freq == 0:
                targetNN.load_state_dict(agentNN.state_dict())

            if done:
                mean_reward = np.sum(episodes_rewards)
                print(f"Episode done with total reward: {mean_reward:.2f}")
                status = 0

        if k < 30:
            epsilon = 1 - 0.9 * (k / 30)
        else:
            epsilon = 0.1


# Testing phase
test = pd.read_csv(r"YNDX_160101_161231.csv")
test = test.rename(columns=columns)
del test["DATE"], test["TIME"], test["VOL"]

test['remove'] = test.apply(lambda row: all([abs(i - row[0]) < 1e-8 for i in row]), axis=1)
test = test.query("remove == False").reset_index(drop=True)
del test['remove']

test["HIGH"] = (test['HIGH'] - test["OPEN"]) / test["OPEN"]
test["LOW"] = (test['LOW'] - test["OPEN"]) / test["OPEN"]
test["CLOSE"] = (test['CLOSE'] - test["OPEN"]) / test["OPEN"]

# Initialize environment for testing
game = Yand(test, commission_perc=0.1, obs_bars=30, test=True)
state1, _, _ = game.step("do_nothing")
state1 = np.array(state1)
state1 = pre_process_state(state1, (1 * input_shape), add_noise=False)
state1 = torch.from_numpy(state1).float()

status = True
step_count = 0
signals = []
have_position = False
entry_price = 0.0
net_profit_loss = 0.0  # cumulative P/L

while status:
    with torch.no_grad():
        q_val = agentNN(state1)
        q_val_ = q_val.numpy()

    action = get_action(q_val_, 3, 0.05) 
    action_name = actions[action]
    print(f"Step: {game.curr_step}, Action: {action_name}, Have Position: {have_position}")

    if action_name == "buy" and not have_position:
        entry_price = game.state["OPEN"].iloc[-1]
        have_position = True
        signals.append({
            "step": game.curr_step,
            "action": "buy",
            "price_open": entry_price,
            "profit_loss_percent": None
        })

    elif action_name == "close" and have_position:
        exit_price = game.state["OPEN"].iloc[-1]
        profit_loss = 100.0 * (exit_price - entry_price) / entry_price
        net_profit_loss += profit_loss
        signals.append({
            "step": game.curr_step,
            "action": "close",
            "price_open": exit_price,
            "profit_loss_percent": profit_loss
        })
        have_position = False

    # Proceed with environment step
    state2, reward, done = game.step(action_name)
    state2 = np.array(state2)
    state2 = pre_process_state(state2, (1 * input_shape), add_noise=False)
    state2 = torch.from_numpy(state2).float()

    state1 = state2
    step_count += 1
    clear_output(wait=True)

    if done:
        print(f"Testing completed in {step_count} steps.")
        print(f"Net profit/loss over the session: {net_profit_loss:.2f}%")
        status = False

# Print signals with profit/loss info
for signal in signals:
    pl_text = f", Profit/Loss: {signal['profit_loss_percent']:.2f}%" if signal['profit_loss_percent'] is not None else ""
    print(f"Step {signal['step']} - {signal['action']} at open price {signal['price_open']}{pl_text}")

