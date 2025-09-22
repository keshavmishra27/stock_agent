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
Q(s,a)=V(s)+A(s,a)− 
∣A∣
1
  
a
∑
 A(s,a)


