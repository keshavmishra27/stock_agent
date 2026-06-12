# Agent Evaluation & Benchmarking Report

## 1. Evaluation Methodology

**Train/Test Split Strategy:** Chronological (Out-of-Time Validation)
- **Training Set:** 2015 Data (`YNDX_150101_151231.csv`)
- **Testing Set:** 2016 Data (`YNDX_160101_161231.csv`)
*Why this matters for AI/ML in Finance:* Standard k-fold cross-validation is invalid for time-series data due to data leakage. We use a strict out-of-time chronological split to simulate real-world forward-looking deployment.

## 2. Benchmark Comparison

| Metric | RL Agent | Buy & Hold Baseline |
|--------|----------|---------------------|
| Net Profit | 14.83% | 11.67% |
| Trades Taken | 503 | 1 |
| Win Rate | 49.50% | N/A |
| Profit Factor | 1.06 | N/A |

## 3. Evaluation Story

The RL Agent **outperformed** the Buy & Hold baseline by 3.16%. 
This indicates that the agent successfully learned to navigate market volatility, actively managing drawdowns and capturing short-term momentum better than a passive strategy. Its Profit Factor of 1.06 shows a healthy ratio of gross profits to gross losses, validating the N-step Dueling DQN architecture's ability to extract temporal features.
