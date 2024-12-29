# Poker hand simulation

A simple testbed to experiment with Python acceleration using `numba` (CPU-jitted code) and `numba.cuda` (GPU CUDA-jitted code).

We estimate poker hand probabilities using Monte Carlo simulation and compare results with known reference values.

We compare execution speeds using Python, `numba`, `numba` with multiprocessing, and `numba.cuda`.

Open the [notebook](https://github.com/hhoppe/poker_hand/blob/main/poker_hand.ipynb) to see the results.

Or [open the notebook in Colab](https://colab.research.google.com/github/hhoppe/poker_hand/blob/main/poker_hand.ipynb),
change the Runtime to have access to a GPU (e.g., "T4 GPU" (Tesla T4 GPU)), and then run the notebook.
