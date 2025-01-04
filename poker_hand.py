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
#   <td><b>My PC</b> Win10</td>
#   <td style="text-align: center">24</td>
#   <td style="text-align: center">GeForce 3080 Ti</td>
#   <td style="text-align: center">80</td>
#   <td style="text-align: right">24,600</td>
#   <td style="text-align: right">10,000,000</td>
#   <td style="text-align: right">-</td>
#   <td style="text-align: right">840,000,000</td>
# </tr>
# <tr>
#   <td><b>My PC</b> WSL2</td>
#   <td style="text-align: center">24</td>
#   <td style="text-align: center">GeForce 3080 Ti</td>
#   <td style="text-align: center">80</td>
#   <td style="text-align: right">116,000</td>
#   <td style="text-align: right">7,390,000</td>
#   <td style="text-align: right">81,500,000</td>
#   <td style="text-align: right">780,000,000</td>
# </tr>
# <tr>
#   <td><b>Marcel PC</b> Win</td>
#   <td style="text-align: center">24</td>
#   <td style="text-align: center">Titan V</td>
#   <td style="text-align: center">80</td>
#   <td style="text-align: right">62,800</td>
#   <td style="text-align: right">5,160,000</td>
#   <td style="text-align: right">-</td>
#   <td style="text-align: right">2,280,000,000</td>
# </tr><tr>
#   <td><a href="https://colab.research.google.com/github/hhoppe/poker_hand/blob/main/poker_hand.ipynb"><b>Colab</b> T4</a></td>
#   <td style="text-align: center">2</td>
#   <td style="text-align: center">Tesla T4</td>
#   <td style="text-align: center">40</td>
#   <td style="text-align: right">14,500</td>
#   <td style="text-align: right">3,980,000</td>
#   <td style="text-align: right">4,270,000</td>
#   <td style="text-align: right">1,660,000,000</td>
# </tr>
# <tr>
#   <td><a href="https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/poker_hand/blob/main/poker_hand.ipynb"><b>Kaggle</b> T4</a></td>
#   <td style="text-align: center">4</td>
#   <td style="text-align: center">Tesla T4 x2</td>
#   <td style="text-align: center">40</td>
#   <td style="text-align: right">17,300</td>
#   <td style="text-align: right">4,100,000</td>
#   <td style="text-align: right">9,130,000</td>
#   <td style="text-align: right">1,820,000,000</td>
# </tr>
# <tr>
#   <td><b>Kaggle</b> P100</td>
#   <td style="text-align: center">4</td>
#   <td style="text-align: center">Tesla P100</td>
#   <td style="text-align: center">56</td>
#   <td style="text-align: right">18,000</td>
#   <td style="text-align: right">4,070,000</td>
#   <td style="text-align: right">9,900,000</td>
#   <td style="text-align: right">1,120,000,000</td>
# </tr>
# <tr>
#   <td><a href="https://mybinder.org/v2/gh/hhoppe/poker_hand/main?urlpath=lab/tree/poker_hand.ipynb"><b>mybinder</b></a></td>
#   <td style="text-align: center">72</td>
#   <td style="text-align: center">None</td>
#   <td style="text-align: center">-</td>
#   <td style="text-align: right">16,000</td>
#   <td style="text-align: right">1,210,000</td>
#   <td style="text-align: right">740,000</td>
#   <td style="text-align: right">-</td>
# </tr>
# <tr>
#   <td><a href="https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fpoker_hand%2Fblob%2Fmain%2Fpoker_hand.ipynb"><b>deepnote</b></a></td>
#   <td style="text-align: center">8</td>
#   <td style="text-align: center">None</td>
#   <td style="text-align: center">-</td>
#   <td style="text-align: right">14,200</td>
#   <td style="text-align: right">3,950,000</td>
#   <td style="text-align: right">2,840,000</td>
#   <td style="text-align: right">-</td>
# </tr>
# </table>

# %% [markdown]
# It is puzzling that the CUDA rate is lower on my PC than on the online servers.

# %% [markdown]
# ### Imports

# %%
# !pip install -q numba

# %%
# import os
# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
# os.environ['NUMBA_CUDA_DEBUGINFO'] = '1'

# %%
import enum
import math
from math import comb
import multiprocessing
import pathlib
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
class Outcome(enum.IntEnum):
  """Poker hand rankings from best to worst."""

  string_name: str

  ROYAL_FLUSH = 0, 'Royal flush'
  STRAIGHT_FLUSH = 1, 'Straight flush'
  FOUR_OF_A_KIND = 2, 'Four of a kind'
  FULL_HOUSE = 3, 'Full house'
  FLUSH = 4, 'Flush'
  STRAIGHT = 5, 'Straight'
  THREE_OF_A_KIND = 6, 'Three of a kind'
  TWO_PAIR = 7, 'Two pair'
  ONE_PAIR = 8, 'One pair'
  HIGH_CARD = 9, 'High card'

  def __new__(cls, value, string_name):
    obj = int.__new__(cls, value)
    obj._value_ = value
    obj.string_name = string_name
    return obj


# %%
DECK_SIZE = 52
NUM_RANKS = 13
HAND_SIZE = 5
HANDS_PER_DECK = DECK_SIZE - HAND_SIZE + 1
NUM_OUTCOMES = len(Outcome)

# %%
REFERENCE_COUNT_OF_OUTCOME = {  # https://en.wikipedia.org/wiki/Poker_probability
    Outcome.ROYAL_FLUSH: comb(4, 1),
    Outcome.STRAIGHT_FLUSH: comb(10, 1) * comb(4, 1) - comb(4, 1),
    Outcome.FOUR_OF_A_KIND: comb(13, 1) * comb(4, 4) * comb(12, 1) * comb(4, 1),
    Outcome.FULL_HOUSE: comb(13, 1) * comb(4, 3) * comb(12, 1) * comb(4, 2),
    Outcome.FLUSH: comb(13, 5) * comb(4, 1) - comb(10, 1) * comb(4, 1),
    Outcome.STRAIGHT: comb(10, 1) * comb(4, 1) ** 5 - comb(10, 1) * comb(4, 1),
    Outcome.THREE_OF_A_KIND: comb(13, 1) * comb(4, 3) * comb(12, 2) * comb(4, 1) ** 2,
    Outcome.TWO_PAIR: comb(13, 2) * comb(4, 2) ** 2 * comb(11, 1) * comb(4, 1),
    Outcome.ONE_PAIR: comb(13, 1) * comb(4, 2) * comb(12, 3) * comb(4, 1) ** 3,
    Outcome.HIGH_CARD: (comb(13, 5) - comb(10, 1)) * (comb(4, 1) ** 5 - comb(4, 1)),
}
EXPECTED_PROB = np.array(list(REFERENCE_COUNT_OF_OUTCOME.values())) / comb(DECK_SIZE, HAND_SIZE)
assert np.allclose(np.sum(EXPECTED_PROB), 1.0)


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
  # pylint: disable=consider-using-in, consider-using-max-builtin

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
  # Expanding `cards` on the rhs results in better cuda-compiled code.
  c0, c1, c2, c3, c4 = cards[0], cards[1], cards[2], cards[3], cards[4]
  is_flush = get_suit(c0) == get_suit(c1) == get_suit(c2) == get_suit(c3) == get_suit(c4)

  # Expanding `ranks` on the rhs results in better cuda-compiled code.
  r0, r1, r2, r3, r4 = ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]
  # We must also consider the ace-low straight 'A2345'.
  is_straight = r1 == r0 + 1 and r2 == r0 + 2 and r3 == r0 + 3 and (r4 == r0 + 4 or r4 == r0 + 12)

  # Count rank frequencies.
  freqs[:] = 0
  for rank in ranks:
    freqs[rank] += 1

  # max_freq = max(freqs)
  # num_pairs = np.sum(freqs == 2)
  max_freq = 0
  num_pairs = 0
  for freq in freqs:
    if freq > max_freq:
      max_freq = freq
    if freq == 2:
      num_pairs += 1

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

  def compute_cpu(num_decks, rng):
    BLOCK_SIZE = 10
    # deck_block = np.arange(DECK_SIZE, dtype=np.uint8).repeat(BLOCK_SIZE).reshape(DECK_SIZE, -1).T
    deck_block = np.empty((BLOCK_SIZE, DECK_SIZE), dtype=np.uint8)
    deck_block[:] = np.arange(DECK_SIZE, dtype=np.uint8)
    ranks = np.empty(HAND_SIZE, np.uint8)
    freqs = np.empty(NUM_RANKS, np.int8)
    tally = np.zeros(NUM_OUTCOMES, np.int64)
    for deck_index_start in range(0, num_decks, BLOCK_SIZE):
      deck_block2 = rng.permutation(deck_block, axis=1)
      for deck_index in range(deck_index_start, min(deck_index_start + BLOCK_SIZE, num_decks)):
        deck = deck_block2[deck_index - deck_index_start]
        for hand_index in range(HANDS_PER_DECK):
          hand = deck[hand_index : hand_index + 5]
          outcome = evaluate_hand(hand, ranks, freqs)
          tally[outcome] += 1
    return tally / (num_decks * HANDS_PER_DECK)

  return compute_cpu


# %%
simulate_hands_cpu_python = make_compute_cpu(evaluate_hand_python)
simulate_hands_cpu_numba = numba.njit(make_compute_cpu(evaluate_hand_numba))


# %%
assert np.allclose(
    simulate_hands_cpu_numba(10**5, np.random.default_rng(1)), EXPECTED_PROB, atol=0.001
)

# %%
# %timeit -n1 -r2 simulate_hands_cpu_python(2000, np.random.default_rng(1))
# %timeit -n100 -r2 simulate_hands_cpu_numba(10**4, np.random.default_rng(1))

# %%
# 708 ms ± 13.2 ms per loop (mean ± std. dev. of 2 runs, 1 loop each)
# 14.8 ms ± 60.2 µs per loop (mean ± std. dev. of 2 runs, 100 loops each)


# %%
def compute_chunk(args):
  num_decks, rng = args
  return simulate_hands_cpu_numba(num_decks, rng)


# %%
def simulate_hands_cpu_numba_multiprocess(num_decks, rng):
  num_processes = multiprocessing.cpu_count()
  chunk_num_decks = math.ceil(num_decks / num_processes)
  chunks = [(chunk_num_decks, np.random.default_rng(rng)) for i in range(num_processes)]
  with multiprocessing.get_context('fork').Pool(num_processes) as pool:
    results = pool.map(compute_chunk, chunks)
  return np.mean(results, axis=0)


# %% [markdown]
# ### GPU simulation


# %%
@cuda.jit(fastmath=True)  # Using cuda.local.array.
def old_compute_gpu(rng_states, num_decks_per_thread, global_tally):
  # pylint: disable=too-many-function-args, no-value-for-parameter, comparison-with-callable
  USE_UINT_RANDOM = True
  thread_index = cuda.grid(1)
  if thread_index >= len(rng_states):
    return

  thread_tally = cuda.local.array(NUM_OUTCOMES, np.int32)
  deck = cuda.local.array(DECK_SIZE, np.uint8)
  ranks = cuda.local.array(HAND_SIZE, np.uint8)
  freqs = cuda.local.array(NUM_RANKS, np.uint8)
  shared_tally = cuda.shared.array(NUM_OUTCOMES, np.int64)  # Per-block intermediate tally.

  thread_tally[:] = 0
  for i in range(DECK_SIZE):
    deck[i] = i

  for _ in range(num_decks_per_thread):
    # Apply Fisher-Yates shuffle to current deck.
    for i in range(51, 0, -1):
      # See https://github.com/numba/numba/blob/main/numba/cuda/random.py
      if USE_UINT_RANDOM:  # Faster and has lower bias (~2.76e-18 for worst case ).
        j = cuda.random.xoroshiro128p_next(rng_states, thread_index) % numba.uint32(i + 1)
      else:  # Results in higher bias (~2.86e-6) due to reduced size (24 bits) of mantissa.
        j = int(cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_index) * (i + 1))
      deck[i], deck[j] = deck[j], deck[i]

    for hand_index in range(HANDS_PER_DECK):
      hand = deck[hand_index : hand_index + 5]
      outcome = evaluate_hand_numba(hand, ranks, freqs)
      thread_tally[outcome] += 1

  # First accumulate a per-block tally, then accumulate that tally into the global tally.
  thread_id = cuda.threadIdx.x
  if thread_id == 0:
    for i in range(NUM_OUTCOMES):
      shared_tally[i] = 0
  cuda.syncthreads()

  # Each thread adds its local results to shared memory.
  for i in range(NUM_OUTCOMES):
    cuda.atomic.add(shared_tally, i, thread_tally[i])
    cuda.syncthreads()

  if thread_id == 0:
    for i in range(NUM_OUTCOMES):
      cuda.atomic.add(global_tally, i, shared_tally[i])


# %%
THREADS_PER_BLOCK = 64


# %%
@cuda.jit(fastmath=True)  # Using cuda.shared.array.
def compute_gpu(rng_states, num_decks_per_thread, global_tally):
  # pylint: disable=too-many-function-args, no-value-for-parameter, comparison-with-callable
  # pylint: disable=possibly-used-before-assignment
  USE_UINT_RANDOM = True
  thread_index = cuda.grid(1)
  if thread_index >= len(rng_states):
    return
  thread_id = cuda.threadIdx.x
  block_tally = cuda.shared.array((THREADS_PER_BLOCK, NUM_OUTCOMES), np.int32)
  block_deck = cuda.shared.array((THREADS_PER_BLOCK, DECK_SIZE), np.uint8)
  block_ranks = cuda.shared.array((THREADS_PER_BLOCK, HAND_SIZE), np.uint8)
  block_freqs = cuda.shared.array((THREADS_PER_BLOCK, NUM_RANKS), np.uint8)
  tally = block_tally[thread_id]
  deck = block_deck[thread_id]
  ranks = block_ranks[thread_id]
  freqs = block_freqs[thread_id]

  tally[:] = 0
  for i in range(numba.uint8(DECK_SIZE)):  # Casting as uint8 nicely unrolls the loop.
    deck[i] = i

  for _ in range(num_decks_per_thread):
    # Apply Fisher-Yates shuffle to current deck.
    for i in range(51, 0, -1):
      # See https://github.com/numba/numba/blob/main/numba/cuda/random.py
      if USE_UINT_RANDOM:  # Faster and has lower bias (~2.76e-18 for worst case ).
        random_uint64 = cuda.random.xoroshiro128p_next(rng_states, thread_index)
        j = numba.uint32(random_uint64) % numba.uint32(i + 1)
      else:  # Results in higher bias (~2.86e-6) due to reduced size (24 bits) of mantissa.
        j = int(cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_index) * (i + 1))
      deck[i], deck[j] = deck[j], deck[i]

    for hand_index in range(HANDS_PER_DECK):
      hand = deck[hand_index : hand_index + 5]
      outcome = evaluate_hand_numba(hand, ranks, freqs)
      tally[outcome] += 1

  # First accumulate a per-block tally, then accumulate that tally into the global tally.
  shared_tally = cuda.shared.array(NUM_OUTCOMES, np.int64)  # Per-block intermediate tally.
  if thread_id == 0:
    shared_tally[:] = 0
  cuda.syncthreads()

  # Each thread adds its local results to shared memory.
  for i in range(NUM_OUTCOMES):
    cuda.atomic.add(shared_tally, i, tally[i])
    cuda.syncthreads()

  if thread_id == 0:
    for i in range(NUM_OUTCOMES):
      cuda.atomic.add(global_tally, i, shared_tally[i])


# %%
def simulate_hands_gpu_cuda(num_decks, rng):
  device = cuda.get_current_device()
  # Target enough threads for ~4 blocks per SM.
  target_num_threads = 4 * device.MULTIPROCESSOR_COUNT * THREADS_PER_BLOCK
  num_decks_per_thread = max(1, num_decks // target_num_threads)
  num_threads = num_decks // num_decks_per_thread
  # print(f'{num_decks_per_thread=} {num_threads=}')
  blocks = math.ceil(num_threads / THREADS_PER_BLOCK)
  seed = rng.integers(2**64, dtype=np.uint64)
  d_rng_states = cuda.random.create_xoroshiro128p_states(num_threads, seed)
  d_global_tally = cuda.to_device(np.zeros(NUM_OUTCOMES, np.int64))
  compute_gpu[blocks, THREADS_PER_BLOCK](d_rng_states, num_decks_per_thread, d_global_tally)
  return d_global_tally.copy_to_host() / (num_threads * num_decks_per_thread * HANDS_PER_DECK)


# %%
assert np.allclose(
    simulate_hands_gpu_cuda(10**7, np.random.default_rng(1)), EXPECTED_PROB, atol=0.0001
)

# %%
# %timeit -n1 -r5 simulate_hands_gpu_cuda(10**7, np.random.default_rng(1))  # ~300 ms.

# %%
# simulate_poker_hands(10**8, 'gpu_cuda', simulate_hands_gpu_cuda)


# %%
def write_cuda_assembly_code():
  if 0:
    signature = (cuda.random.xoroshiro128p_type[:], numba.int64, numba.int64[:])
    ptx, _ = cuda.compile_ptx_for_current_device(compute_gpu, signature, device=True, abi='c')
  else:
    (ptx,) = compute_gpu.inspect_asm().values()
  pathlib.Path('compute_gpu.ptx').write_text(ptx, encoding='utf-8')


# %%
def report_kernel_properties():
  PROPERTIES = 'const_mem_size local_mem_per_thread max_threads_per_block regs_per_thread shared_mem_per_block'.split()
  for property_name in PROPERTIES:
    (value,) = getattr(compute_gpu, 'get_' + property_name)().values()
    print(f'{property_name} = {value}')


# %%
if cuda.is_available():
  write_cuda_assembly_code()
  report_kernel_properties()

# %%
# const_mem_size = 0
# local_mem_per_thread = 112
# max_threads_per_block = 1024
# regs_per_thread = 37
# shared_mem_per_block = 80

# const_mem_size = 0
# local_mem_per_thread = 0
# max_threads_per_block = 1024
# regs_per_thread = 40
# shared_mem_per_block = 7120

# %% [markdown]
# ### Results

# %%
SIMULATE_FUNCTIONS: Any = {
    'cpu_python': simulate_hands_cpu_python,
    'cpu_numba': simulate_hands_cpu_numba,
}
if 'fork' in multiprocessing.get_all_start_methods():
  SIMULATE_FUNCTIONS.update({'cpu_numba_multiprocess': simulate_hands_cpu_numba_multiprocess})
if cuda.is_available():
  SIMULATE_FUNCTIONS.update({'gpu_cuda': simulate_hands_gpu_cuda})

# %%
COMPLEXITY_ADJUSTMENT = {
    'cpu_python': 0.01,
    'cpu_numba': 2.0,
    'cpu_numba_multiprocess': 20.0,
    'gpu_cuda': 100.0,
}


# %%
def simulate_poker_hands(base_num_hands, func_name, func, seed=1):
  base_num_decks = math.ceil(base_num_hands / HANDS_PER_DECK)
  num_decks = math.ceil(base_num_decks * COMPLEXITY_ADJUSTMENT[func_name])
  num_hands = num_decks * HANDS_PER_DECK
  print(f'\nFor {func_name} simulating {num_hands:,} hands:')

  # Ensure the function is jitted.
  _ = func(int(100_000 * COMPLEXITY_ADJUSTMENT[func_name]), np.random.default_rng(seed))

  start_time = time.perf_counter_ns()
  results = func(num_decks, np.random.default_rng(seed))
  elapsed_time = (time.perf_counter_ns() - start_time) / 10**9

  round_digits = lambda x, ndigits=3: round(x, ndigits - 1 - math.floor(math.log10(abs(x))))
  hands_per_s = round_digits(int(num_hands / elapsed_time))
  print(f' Elapsed time is {elapsed_time:.3f} s, or {hands_per_s:,} hands/s.')

  print(' Probabilities:')
  for (outcome, reference_count), prob in zip(REFERENCE_COUNT_OF_OUTCOME.items(), results):
    reference_prob = reference_count / comb(DECK_SIZE, HAND_SIZE)
    error = prob - reference_prob
    s = f'  {outcome.string_name:<16}: {prob * 100:8.5f}%'
    s += f'  (vs. reference {reference_prob * 100:8.5f}%  error:{error * 100:8.5f}%)'
    print(s)


# %%
def compare_simulations(base_num_hands, seed=1):
  for func_name, func in SIMULATE_FUNCTIONS.items():
    simulate_poker_hands(base_num_hands, func_name, func, seed)


# %%
compare_simulations(base_num_hands=10**7)

# %%
# 116k, 7.4m, 82m, 780m
# 123k, 16m, 156m, 1400m
# 137k, 32m, 340m, 2000m

# %% [markdown]
# ### End
