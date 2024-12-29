# %% [markdown]
# # Poker hand simulation
#
# A simple testbed to experiment with Python acceleration using `numba` (CPU-jitted code) and `numba.cuda` (GPU CUDA-jitted code).
#
# We estimate poker hand probabilities using Monte Carlo simulation and compare results with known reference values.
#
# We compare execution speeds using Python, `numba`, `numba` with multiprocessing, and `numba.cuda`.
#
# One way to run this notebook is to
# [open it in Colab](https://colab.research.google.com/github/hhoppe/poker_hand/blob/main/poker_hand.ipynb),
# change the Runtime to have access to a GPU (e.g., "T4 GPU" (Tesla T4 GPU)), and then "run all cells".

# %% [markdown]
# ### Imports

# %%
import math
import multiprocessing
import time
from typing import Any

import numba
from numba import cuda
import numba.cuda.random
import numpy as np


# %%
cuda.detect()


# %% [markdown]
# ### Hand evaluation


# %%
def evaluate_hand_python(cards, ranks, freqs):

  def get_rank(card):
    return card >> 2  # 13 ranks numbered 0-12 (23456789TJQKA).

  def get_suit(card):
    return card & 0b11  # 4 suits numbered 0-3.

  # Sort cards by rank for easier pattern matching, using simple insertion sort.
  for i in range(5):
    ranks[i] = get_rank(cards[i])
  for i in range(1, 5):
    key = ranks[i]
    j = i - 1
    while j >= 0 and ranks[j] > key:
      ranks[j + 1] = ranks[j]
      j -= 1
    ranks[j + 1] = key

  c0, c1, c2, c3, c4 = cards[0], cards[1], cards[2], cards[3], cards[4]
  is_flush = get_suit(c0) == get_suit(c1) == get_suit(c2) == get_suit(c3) == get_suit(c4)
  r0, r1, r2, r3, r4 = ranks
  is_straight = r1 == r0 + 1 and r2 == r0 + 2 and r3 == r0 + 3 and r4 - r0 in (4, 12)

  # Count rank frequencies.
  freqs[:] = 0
  for rank in ranks:
    freqs[rank] += 1
  # num_pairs = np.sum(freqs == 2)
  num_pairs = 0
  for i in range(13):
    if freqs[i] == 2:
      num_pairs += 1
  max_freq = max(freqs)

  if is_flush and is_straight:
    return 0 if ranks[0] == 8 else 1  # Royal flush or straight flush.
  if max_freq == 4:
    return 2  # Four of a kind.
  if max_freq == 3 and num_pairs > 0:
    return 3  # Full house.
  if is_flush:
    return 4  # Flush.
  if is_straight:
    return 5  # Straight.
  if max_freq == 3:
    return 6  # Three of a kind.
  if max_freq == 2:
    if num_pairs == 2:
      return 7  # Two pair.
    return 8  # One pair.
  return 9  # High card.


# %%
evaluate_hand_numba = numba.njit(evaluate_hand_python)

# %% [markdown]
# ### CPU simulation


# %%
def make_compute_cpu(evaluate_hand):
  # Return a specialized compute function for the given evaluate_hand function.

  def compute(num_decks, seed):
    np.random.seed(seed)
    deck = np.arange(52, dtype=np.uint8)
    ranks = np.empty(5, np.uint8)
    freqs = np.empty(13, np.int8)
    tally = np.zeros(10, np.int64)
    for _ in range(num_decks):
      np.random.shuffle(deck)
      for hand_index in range(10):
        hand = deck[hand_index * 5 : (hand_index + 1) * 5]
        outcome = evaluate_hand(hand, ranks, freqs)
        tally[outcome] += 1
    return tally

  return compute


compute_cpu_python = make_compute_cpu(evaluate_hand_python)
compute_cpu_numba = numba.njit(make_compute_cpu(evaluate_hand_numba))


# %%
def compute_chunk(args):
  num_decks, seed = args
  return compute_cpu_numba(num_decks, seed)


# %%
def run_cpu_python(num_decks, seed=1):
  return compute_cpu_python(num_decks, seed=seed) / (num_decks * 10)


# %%
def run_cpu_numba(num_decks, seed=1):
  return compute_cpu_numba(num_decks, seed=seed) / (num_decks * 10)


# %%
def run_cpu_numba_multiprocess(num_decks, seed=1):
  num_processes = multiprocessing.cpu_count()
  chunk_num_decks = math.ceil(num_decks / num_processes)
  base_seed = seed * 10_000_000
  chunks = [(chunk_num_decks, base_seed + i) for i in range(num_processes)]
  # with multiprocessing.get_context('fork').Pool(num_processes) as pool:
  with multiprocessing.Pool(num_processes) as pool:
    results = pool.map(compute_chunk, chunks)
  return np.sum(results, axis=0) / (num_processes * chunk_num_decks * 10)


# %% [markdown]
# ### GPU simulation


# %%
@cuda.jit
def compute_gpu(rng_states, num_decks_per_thread, num_threads, results):
  thread_index = cuda.grid(1)  # pylint: disable=no-value-for-parameter
  if thread_index >= num_threads:
    return

  # Local tally for this thread.
  local_tally = cuda.local.array(10, np.int32)
  for i in range(10):
    local_tally[i] = 0

  deck = cuda.local.array(52, np.uint8)
  ranks = cuda.local.array(5, np.uint8)
  freqs = cuda.local.array(13, np.uint8)

  for _ in range(num_decks_per_thread):
    for i in range(52):
      deck[i] = i
    # Apply Fisher-Yates shuffle to deck.
    for i in range(51, 0, -1):
      j = int(cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_index) * (i + 1))
      deck[i], deck[j] = deck[j], deck[i]

    for hand_index in range(10):
      hand = deck[hand_index * 5 : (hand_index + 1) * 5]
      outcome = evaluate_hand_numba(hand, ranks, freqs)
      local_tally[outcome] += 1

  # Accumulate local tallies in global results.
  for i in range(10):
    cuda.atomic.add(results, i, local_tally[i])  # pylint: disable=too-many-function-args


# %%
def run_gpu_cuda(num_decks, seed=1, num_decks_per_thread=10, threads_per_block=256):
  num_threads = num_decks // num_decks_per_thread
  blocks = math.ceil(num_threads / threads_per_block)
  d_rng_states = cuda.random.create_xoroshiro128p_states(num_threads, seed=seed)
  d_results = cuda.to_device(np.zeros(10, np.int64))
  compute_gpu[blocks, threads_per_block](d_rng_states, num_decks_per_thread, num_threads, d_results)
  return d_results.copy_to_host() / (num_threads * num_decks_per_thread * 10)


# %% [markdown]
# ### Results

# %%
methods: Any = {
    'cpu_python': run_cpu_python,
    'cpu_numba': run_cpu_numba,
    'cpu_numba_multiprocess': run_cpu_numba_multiprocess,
    'gpu_cuda': run_gpu_cuda,
}

# %%
adjust_run_complexity = {
    'cpu_python': 0.025,
    'cpu_numba': 1.0,
    'cpu_numba_multiprocess': 10.0,
    'gpu_cuda': 100.0,
}


# %%
def perform_initial_jits(debug=False, base_num_decks=100_000):
  for processor, func in methods.items():
    num_decks = int(base_num_decks * adjust_run_complexity[processor])
    probs = func(num_decks)
    if debug:
      print(probs[-4:])


# %%
perform_initial_jits()


# %%
def simulate_poker_hands(base_num_hands, seed=1):
  num_hands_per_deck = 10
  base_num_decks = base_num_hands // num_hands_per_deck

  for processor, func in methods.items():
    num_decks = math.ceil(base_num_decks * adjust_run_complexity[processor])
    num_hands = num_decks * 10
    print(f'For {processor} simulating {num_hands:_} hands:')

    start_time = time.monotonic()
    results = func(num_decks, seed=seed)
    elapsed_time = time.monotonic() - start_time

    hands_per_s = int(num_hands / elapsed_time)
    print(f' Elapsed time is {elapsed_time:.3f} s, or {hands_per_s:_} hands/s.')

    comb = math.comb
    reference = {  # https://en.wikipedia.org/wiki/Poker_probability
        'Royal flush': comb(4, 1),
        'Straight flush': comb(10, 1) * comb(4, 1) - comb(4, 1),
        'Four of a kind': comb(13, 1) * comb(4, 4) * comb(12, 1) * comb(4, 1),
        'Full house': comb(13, 1) * comb(4, 3) * comb(12, 1) * comb(4, 2),
        'Flush': comb(13, 5) * comb(4, 1) - comb(10, 1) * comb(4, 1),
        'Straight': comb(10, 1) * comb(4, 1) ** 5 - comb(10, 1) * comb(4, 1),
        'Three of a kind': comb(13, 1) * comb(4, 3) * comb(12, 2) * comb(4, 1) ** 2,
        'Two pair': comb(13, 2) * comb(4, 2) ** 2 * comb(11, 1) * comb(4, 1),
        'One pair': comb(13, 1) * comb(4, 2) * comb(12, 3) * comb(4, 1) ** 3,
        'High card': (comb(13, 5) - comb(10, 1)) * (comb(4, 1) ** 5 - comb(4, 1)),
    }
    assert sum(reference.values()) == comb(52, 5)

    print(' Probabilities:')
    for (hand_name, reference_num_hands), prob in zip(reference.items(), results):
      reference_prob = reference_num_hands / comb(52, 5)
      error = prob - reference_prob
      s = f'  {hand_name:<16}: {prob * 100:8.5f}%'
      s += f'  (vs. reference {reference_prob * 100:8.5f}%  error:{error * 100:8.5f}%)'
      print(s)

    print()


# %%
simulate_poker_hands(base_num_hands=10**7)

# %% [markdown]
# ### End
