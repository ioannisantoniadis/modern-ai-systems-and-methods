# Sequence, Time-Series, and State Models

## Core Idea

Sequence models handle ordered data. State models assume observed behavior is driven by an evolving state, sometimes hidden.

## Markov Chains

A Markov chain models transitions between states:

```text
current state -> next state
```

The Markov assumption says the future depends on the present state, not the full history.

## Hidden Markov Models

An HMM has hidden states and observed emissions:

```text
hidden state_t -> hidden state_t+1
hidden state_t -> observation_t
```

Use when observations are noisy signals of an underlying state.

## State-Space Models

State-space models generalize hidden-state dynamics, often with continuous states. Kalman filters are the classic linear-Gaussian example.

## Neural Sequence Models

- RNNs and LSTMs process sequences recurrently.
- Transformers use attention to model relationships across positions.
- Temporal convolution models use causal convolutions over time.

## Time-Series Topics

- Trend and seasonality.
- Forecasting horizons.
- Autocorrelation.
- Exogenous variables.
- Backtesting.
- Forecast residual anomaly detection.

## Interview Check

Be able to explain when a Markov model, HMM, LSTM, or transformer is appropriate. The decision depends on hidden state, sequence length, data scale, interpretability, and serving constraints.
