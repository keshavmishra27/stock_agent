#**Reinforcement Learning Stock Trading (Dueling DQN with N-Step Returns)**
<br>
This project implements a Deep Q-Network (DQN) with N-step returns and a Dueling Conv1D Q-Network architecture to simulate and backtest stock trading strategies.
The agent learns to take buy, close, or do nothing actions based on time-series stock data, aiming to maximize profit under transaction costs (commissions).



#**Features**
<br>
Custom trading environment (Yand) with:
OHLC stock price preprocessing
Position management (buy/close/hold)
Commission penalties
Reward calculation based on profit/loss percentage
N-Step Replay Buffer for stabilizing training
Dueling DQN architecture using Conv1D layers to process time-series signals
Target network synchronization for stable learning
ε-greedy exploration strategy with annealing
Training and testing loops for live simulation on unseen stock data
Profit and loss tracking with trading signals output
<br>

#**data**
<br>
The project uses Yandex (YNDX) stock tick data in CSV format.
<br>
#**Training dataset:**
YNDX_150101_151231.csv
<br>

#**Testing dataset:**
YNDX_160101_161231.csv
<br>

#**Columns are renamed to standard OHLCV:**
<br>
DATE
TIME
OPEN
HIGH
LOW
CLOSE
VOL
<br>

#**During preprocessing:**
<br>

HIGH, LOW, and CLOSE are normalized relative to the open price.
Rows with constant values are dropped.
Only OHLC values are retained.
<br>

#**Actions**
<br>
The agent can perform 3 actions:
do_nothing → hold position
buy → open a long position
close → close an active long position
<br>

#**Neural Network (Dueling Conv1D Q-Network)
The model uses temporal convolution to extract features across observation bars:
Conv1D layers → extract short-term patterns
Dueling architecture:
State Value stream
Advantage stream
Combined Q-value:
<br>

<pre><code>
Q(s,a)=V(s)+A(s,a)− 1/∣A∣ ∑A(s,a)
                         a
</code></pre>
This structure allows the network to better distinguish the inherent value of the state from the relative advantage of actions.
<br>

#**Training**
N-step replay buffer with discounted reward accumulation
Minibatch size: 30
Memory size: 100
Optimizer: RMSprop
Loss: MSE Loss
Discount factor (γ=>(gamma)): 0.99
Target sync frequency: 10
Epsilon-greedy exploration:
Starts at 1.0
Decays to 0.1
Training runs for a series of episodes until 300 environment steps are completed.
Rewards, losses, and epsilon values are logged.

#**Testing**
The model is tested on unseen data (YNDX_160101_161231.csv).
During testing:
ε is fixed to a low exploration value (0.05)
The model outputs trading signals with prices and profit/loss percentages
Net cumulative profit/loss is displayed at the end
Example output during testing:
<pre><code>
Step 1200, Action: buy, Have Position: False
Step 1205, Action: close, Have Position: True
Testing completed in 500 steps.
Net profit/loss over the session: 12.35%
</code></pre>
<br>

#**Signals summary:**
<pre><code>
Step 1200 - buy at open price 80.25
Step 1205 - close at open price 81.30, Profit/Loss: 1.31%
</code></pre>
<br>

# **project structure**
<pre><code>
📂 project/
 ├── YNDX_150101_151231.csv   # Training dataset
 ├── YNDX_160101_161231.csv   # Testing dataset
 ├── train_test_dqn.py        # Main implementation (the code provided)
 ├── README.md                # Documentation (this file)
</code></pre>
<br>

#**Dependencies**
<br>
Python 3.8+
NumPy
Pandas
PyTorch
IPython (for clear_output)
Matplotlib/Seaborn (optional for plotting)
<br>

#**Install requirements:**
<br>
<pre><code>
  pip install numpy pandas torch ipython matplotlib
</code></pre>
<br>

#**How to Run**
<br>
Place the training and testing CSVs (YNDX_150101_151231.csv, YNDX_160101_161231.csv) in the project folder.
Run the script:
<pre><code>
  python train_test_dqn.py
</code></pre>
<br>




