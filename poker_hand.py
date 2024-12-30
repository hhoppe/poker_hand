# %% [markdown]
# # Poker hand simulation
#
# This notebook is a simple testbed to experiment with Python acceleration using `numba` (CPU-jitted code) and `numba.cuda` (GPU CUDA-jitted code).
#
# We estimate poker hand probabilities using Monte Carlo simulation and compare results with known reference values.
#
# We compare execution speeds using Python, `numba`, `numba` with multiprocessing, and `numba.cuda`.
#
# Websites on which to run this notebook include:
#
# - [Open in Google Colab](https://colab.research.google.com/github/hhoppe/poker_hand/blob/main/poker_hand.ipynb), click on Runtime -> Change runtime type -> T4 GPU, and then Runtime -> Run all.
#
# - [Open in kaggle.com](https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/poker_hand/blob/main/poker_hand.ipynb), click on Session options -> Accelerator -> GPU T4 x2 or P100, then Run All.
#
# - [Open in mybinder.org](https://mybinder.org/v2/gh/hhoppe/poker_hand/main?urlpath=lab/tree/poker_hand.ipynb).  Unfortunately, no GPU is available.
#
# - [Open in deepnote.com](https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fpoker_hand%2Fblob%2Fmain%2Fpoker_hand.ipynb).  Unfortunately, no GPU is available.
#
# Here are results:

# %% [markdown]
# <table style="margin-left: 0">
# <tr>
#   <th>Platform</th>
#   <th style="text-align: center">CPU<br>threads</th>
#   <th style="text-align: center">GPU<br>type</th>
#   <th style="text-align: center">CUDA<br>SMs</th>
#   <th colspan="4">Simulation rate (hands/s)</th>
# </tr>
# <tr>
#   <th></th>
#   <th></th>
#   <th></th>
#   <th></th>
#   <th>Python</th>
#   <th>Numba</th>
#   <th>Multiprocess</th>
#   <th>CUDA</th>
# </tr>
# <tr>
#   <td><b>My PC</b> WSL2</td>
#   <td style="text-align: center">24</td>
#   <td style="text-align: center">GeForce 3080 Ti</td>
#   <td style="text-align: center">80</td>
#   <td style="text-align: right">115,575</td>
#   <td style="text-align: right">7,391,506</td>
#   <td style="text-align: right">81,486,454</td>
#   <td style="text-align: right">780,537,776</td>
# </tr>
# <tr>
#   <td><a href="https://colab.research.google.com/github/hhoppe/poker_hand/blob/main/poker_hand.ipynb"><b>Colab</b> T4</a></td>
#   <td style="text-align: center">2</td>
#   <td style="text-align: center">Tesla T4</td>
#   <td style="text-align: center">40</td>
#   <td style="text-align: right">14,537</td>
#   <td style="text-align: right">3,979,992</td>
#   <td style="text-align: right">4,265,099</td>
#   <td style="text-align: right">1,656,967,205</td>
# </tr>
# <tr>
#   <td><a href="https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/poker_hand/blob/main/poker_hand.ipynb"><b>Kaggle</b> T4</a></td>
#   <td style="text-align: center">4</td>
#   <td style="text-align: center">Tesla T4 x2</td>
#   <td style="text-align: center">40</td>
#   <td style="text-align: right">17,333</td>
#   <td style="text-align: right">4,095,869</td>
#   <td style="text-align: right">9,132,356</td>
#   <td style="text-align: right">1,820,619,723</td>
# </tr>
# <tr>
#   <td><b>Kaggle</b> P100</td>
#   <td style="text-align: center">4</td>
#   <td style="text-align: center">Tesla P100</td>
#   <td style="text-align: center">56</td>
#   <td style="text-align: right">18,007</td>
#   <td style="text-align: right">4,068,247</td>
#   <td style="text-align: right">9,895,509</td>
#   <td style="text-align: right">1,115,351,475</td>
# </tr>
# <tr>
#   <td><a href="https://mybinder.org/v2/gh/hhoppe/poker_hand/main?urlpath=lab/tree/poker_hand.ipynb"><b>mybinder</b></a></td>
#   <td style="text-align: center">72</td>
#   <td style="text-align: center">None</td>
#   <td style="text-align: center">-</td>
#   <td style="text-align: right">15,954</td>
#   <td style="text-align: right">1,206,560</td>
#   <td style="text-align: right">743,016</td>
#   <td style="text-align: right">-</td>
# </tr>
# <tr>
#   <td><a href="https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fpoker_hand%2Fblob%2Fmain%2Fpoker_hand.ipynb"><b>deepnote</b></a></td>
#   <td style="text-align: center">8</td>
#   <td style="text-align: center">None</td>
#   <td style="text-align: center">-</td>
#   <td style="text-align: right">14,245</td>
#   <td style="text-align: right">3,953,857</td>
#   <td style="text-align: right">2,842,231</td>
#   <td style="text-align: right">-</td>
# </tr>
# </table>

# %% [markdown]
# It is puzzling that the CUDA rate is lower on my PC (given its good GPU).<br/>
# The use of "Windows Subsystem for Linux" might be introducing some overhead.

# %% [markdown]
# ### Imports

# %%
# %pip install -q numba

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
if cuda.is_available() and cuda.detect():
  print(f'The number of GPU SMs is {cuda.get_current_device().MULTIPROCESSOR_COUNT}')


# %% [markdown]
# ### Hand evaluation


# %%
DECK_SIZE = 52
NUM_RANKS = 13
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
    return card >> 2  # In range(NUM_RANKS), ordered '23456789TJQKA'.

  def get_suit(card: int) -> int:
    return card & 0b11  # In range(4).

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
    if ranks[0] == 8:
      return Outcome.ROYAL_FLUSH.value
    return Outcome.STRAIGHT_FLUSH.value
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
def simulate_hands_gpu_cuda(num_decks, seed, threads_per_block=64):
  device = cuda.get_current_device()
  # Target enough threads for ~4 blocks per SM.
  target_num_threads = 4 * device.MULTIPROCESSOR_COUNT * threads_per_block
  num_decks_per_thread = max(1, num_decks // target_num_threads)
  num_threads = num_decks // num_decks_per_thread
  # print(f'{num_decks_per_thread=} {num_threads=}')
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
def simulate_poker_hands(base_num_hands, seed=1):
  num_hands_per_deck = 10
  base_num_decks = base_num_hands // num_hands_per_deck

  for func_name, func in SIMULATE_FUNCTIONS.items():
    num_decks = math.ceil(base_num_decks * COMPLEXITY_ADJUSTMENT[func_name])
    num_hands = num_decks * 10
    print(f'\nFor {func_name} simulating {num_hands:_} hands:')

    _ = func(int(100_000 * COMPLEXITY_ADJUSTMENT[func_name]), 1)  # Ensure the function is jitted.
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
