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
import enum
import math
import multiprocessing
import time
from typing import Any

import numba
from numba import cuda
import numba.cuda.random
import numpy as np


# %%
print(f'The number of CPU threads is {multiprocessing.cpu_count()}.')

# %%
cuda.detect()


# %% [markdown]
# ### Hand evaluation


# %%
DECK_SIZE = 52
NUM_RANKS = 13
NUM_SUITS = 4
HAND_SIZE = 5


# %%
class Outcome(enum.IntEnum):
  """Poker hand rankings from best to worst."""
  ROYAL_FLUSH = 0
  STRAIGHT_FLUSH = 1
  FOUR_OF_A_KIND = 2
  FULL_HOUSE = 3
  FLUSH = 4
  STRAIGHT = 5
  THREE_OF_A_KIND = 6
  TWO_PAIR = 7
  ONE_PAIR = 8
  HIGH_CARD = 9


# %%
def evaluate_hand_python(cards, ranks, freqs):
  """Evaluate 5-card poker hand and return outcome ranking.

  Args:
    cards: List of 5 integers representing cards (0-51).
    ranks: Pre-allocated array for storing sorted ranks.
    freqs: Pre-allocated array for counting rank frequencies.

  Returns:
    Integer representing hand ranking (see Outcome enum).
  """

  def get_rank(card: int) -> int:
    return card >> 2  # in range(NUM_RANKS), ordered '23456789TJQKA'.

  def get_suit(card: int) -> int:
    return card & 0b11  # in range(NUM_SUITS).

  # Sort cards by rank for easier pattern matching, using simple insertion sort.
  for i in range(HAND_SIZE):
    ranks[i] = get_rank(cards[i])
  # ranks.sort()
  for i in range(1, HAND_SIZE):
    key = ranks[i]
    j = i - 1
    while j >= 0 and ranks[j] > key:
      ranks[j + 1] = ranks[j]
      j -= 1
    ranks[j + 1] = key

  # is_flush = all(card == cards[0] for card in cards[1:])
  c0, c1, c2, c3, c4 = cards[0], cards[1], cards[2], cards[3], cards[4]
  is_flush = get_suit(c0) == get_suit(c1) == get_suit(c2) == get_suit(c3) == get_suit(c4)
  r0, r1, r2, r3, r4 = ranks
  is_straight = r1 - r0 == r2 - r1 == r3 - r2 == 1 and r4 - r3 in (1, 9)  # 9 for ace-low 'A2345'.

  # Count rank frequencies.
  freqs[:] = 0
  for rank in ranks:
    freqs[rank] += 1
  # num_pairs = np.sum(freqs == 2)
  num_pairs = 0
  for i in range(NUM_RANKS):
    if freqs[i] == 2:
      num_pairs += 1
  max_freq = max(freqs)

  if is_flush and is_straight:
    return Outcome.ROYAL_FLUSH.value if ranks[0] == 8 else Outcome.STRAIGHT_FLUSH.value
  if max_freq == 4:
    return Outcome.FOUR_OF_A_KIND.value
  if max_freq == 3 and num_pairs > 0:
    return Outcome.FULL_HOUSE.value
  if is_flush:
    return Outcome.FLUSH.value
  if is_straight:
    return Outcome.STRAIGHT.value
  if max_freq == 3:
    return Outcome.THREE_OF_A_KIND.value
  if max_freq == 2:
    if num_pairs == 2:
      return Outcome.TWO_PAIR.value
    return Outcome.ONE_PAIR.value
  return Outcome.HIGH_CARD.value


# %%
evaluate_hand_numba = numba.njit(evaluate_hand_python)

# %% [markdown]
# ### CPU simulation


# %%
def make_compute_cpu(evaluate_hand):
  # Return a specialized compute_cpu function for the given evaluate_hand function.

  def compute_cpu(num_decks, seed):
    np.random.seed(seed)
    deck = np.arange(DECK_SIZE, dtype=np.uint8)
    ranks = np.empty(HAND_SIZE, np.uint8)
    freqs = np.empty(NUM_RANKS, np.int8)
    tally = np.zeros(10, np.int64)
    for _ in range(num_decks):
      np.random.shuffle(deck)
      for hand_index in range(10):
        hand = deck[hand_index * HAND_SIZE : (hand_index + 1) * HAND_SIZE]
        outcome = evaluate_hand(hand, ranks, freqs)
        tally[outcome] += 1
    return tally

  return compute_cpu


# %%
compute_cpu_python = make_compute_cpu(evaluate_hand_python)
compute_cpu_numba = numba.njit(make_compute_cpu(evaluate_hand_numba))


# %%
def compute_chunk(args):
  num_decks, seed = args
  return compute_cpu_numba(num_decks, seed)


# %%
def simulate_hands_cpu_python(num_decks, seed):
  return compute_cpu_python(num_decks, seed) / (num_decks * 10)


# %%
def simulate_hands_cpu_numba(num_decks, seed):
  return compute_cpu_numba(num_decks, seed) / (num_decks * 10)


# %%
def simulate_hands_cpu_numba_multiprocess(num_decks, seed):
  num_processes = multiprocessing.cpu_count()
  chunk_num_decks = math.ceil(num_decks / num_processes)
  base_seed = seed * 10_000_000
  chunks = [(chunk_num_decks, base_seed + i) for i in range(num_processes)]
  with multiprocessing.Pool(num_processes) as pool:
    results = pool.map(compute_chunk, chunks)
  return np.sum(results, axis=0) / (num_processes * chunk_num_decks * 10)


# %% [markdown]
# ### GPU simulation


# %%
@cuda.jit
def compute_gpu(rng_states, num_decks_per_thread, results):
  thread_index = cuda.grid(1)  # pylint: disable=no-value-for-parameter
  if thread_index >= len(rng_states):
    return

  local_tally = cuda.local.array(10, np.int32)
  deck = cuda.local.array(DECK_SIZE, np.uint8)
  ranks = cuda.local.array(HAND_SIZE, np.uint8)
  freqs = cuda.local.array(NUM_RANKS, np.uint8)
  local_tally[:] = 0
  for i in range(DECK_SIZE):
    deck[i] = i

  for _ in range(num_decks_per_thread):
    # Apply Fisher-Yates shuffle to current deck.
    for i in range(51, 0, -1):
      # j = cuda.random.xoroshiro128p_next(rng_states, thread_index) % (i + 1)  # Undocumented.
      j = int(cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_index) * (i + 1))
      deck[i], deck[j] = deck[j], deck[i]

    for hand_index in range(10):
      hand = deck[hand_index * HAND_SIZE : (hand_index + 1) * HAND_SIZE]
      outcome = evaluate_hand_numba(hand, ranks, freqs)
      local_tally[outcome] += 1

  # Accumulate local tallies into global results.
  for i in range(10):
    cuda.atomic.add(results, i, local_tally[i])  # pylint: disable=too-many-function-args


# %%
def simulate_hands_gpu_cuda(num_decks, seed, num_decks_per_thread=100, threads_per_block=64):
  num_threads = num_decks // num_decks_per_thread
  blocks = math.ceil(num_threads / threads_per_block)
  d_rng_states = cuda.random.create_xoroshiro128p_states(num_threads, seed)
  d_results = cuda.to_device(np.zeros(10, np.int64))
  compute_gpu[blocks, threads_per_block](d_rng_states, num_decks_per_thread, d_results)
  return d_results.copy_to_host() / (num_threads * num_decks_per_thread * 10)


# %% [markdown]
# ### Results

# %%
SIMULATE_FUNCTIONS: Any = {
    'cpu_python': simulate_hands_cpu_python,
    'cpu_numba': simulate_hands_cpu_numba,
    'cpu_numba_multiprocess': simulate_hands_cpu_numba_multiprocess,
    'gpu_cuda': simulate_hands_gpu_cuda,
}

# %%
COMPLEXITY_ADJUSTMENT = {
    'cpu_python': 0.01,
    'cpu_numba': 1.0,
    'cpu_numba_multiprocess': 5.0,
    'gpu_cuda': 100.0,
}


# %%
def ensure_functions_are_jitted(base_num_decks=100_000):
  for func_name, func in SIMULATE_FUNCTIONS.items():
    num_decks = int(base_num_decks * COMPLEXITY_ADJUSTMENT[func_name])
    _ = func(num_decks, 1)


# %%
def simulate_poker_hands(base_num_hands, seed=1):
  ensure_functions_are_jitted()
  num_hands_per_deck = 10
  base_num_decks = base_num_hands // num_hands_per_deck

  for func_name, func in SIMULATE_FUNCTIONS.items():
    if 'cuda' in func_name and not cuda.is_available():
      continue
    num_decks = math.ceil(base_num_decks * COMPLEXITY_ADJUSTMENT[func_name])
    num_hands = num_decks * 10
    print(f'\nFor {func_name} simulating {num_hands:_} hands:')

    start_time = time.monotonic()
    results = func(num_decks, seed)
    elapsed_time = time.monotonic() - start_time

    hands_per_s = int(num_hands / elapsed_time)
    print(f' Elapsed time is {elapsed_time:.3f} s, or {hands_per_s:_} hands/s.')

    comb = math.comb
    REFERENCE = {  # https://en.wikipedia.org/wiki/Poker_probability
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
    assert sum(REFERENCE.values()) == comb(DECK_SIZE, HAND_SIZE)

    print(' Probabilities:')
    for (outcome_name, reference_num_hands), prob in zip(REFERENCE.items(), results):
      reference_prob = reference_num_hands / comb(DECK_SIZE, HAND_SIZE)
      error = prob - reference_prob
      s = f'  {outcome_name:<16}: {prob * 100:8.5f}%'
      s += f'  (vs. reference {reference_prob * 100:8.5f}%  error:{error * 100:8.5f}%)'
      print(s)


# %%
simulate_poker_hands(base_num_hands=10**7)

# %% [markdown]
# ### End
