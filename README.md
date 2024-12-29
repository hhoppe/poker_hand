# Poker hand simulation

A simple testbed to experiment with Python acceleration using `numba` (CPU-jitted code) and `numba.cuda` (GPU CUDA-jitted code).

We estimate poker hand probabilities using Monte Carlo simulation and compare results with known reference values.

We compare execution speeds using Python, `numba`, `numba` with multiprocessing, and `numba.cuda`.

[**[Open this notebook in Colab]**](https://colab.research.google.com/github/hhoppe/poker_hand/blob/main/poker_hand.ipynb).
