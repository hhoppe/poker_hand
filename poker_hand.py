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
# ## Imports

# %%
# !pip install -q numba

# %%
# !if [ ! -f random32.py ]; then wget https://github.com/hhoppe/poker_hand/raw/main/random32.py; fi

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
import random32


# %%
USE_RANDOM32 = True
if USE_RANDOM32:
  random_create_states = random32.create_xoshiro128p_states
  random_next_uniform_uint = random32.xoshiro128p_next
  random_next_uniform_float32 = random32.xoshiro128p_uniform_float32
else:
  # See https://github.com/numba/numba/blob/main/numba/cuda/random.py
  random_create_states = cuda.random.create_xoroshiro128p_states
  random_next_uniform_uint = cuda.random.xoroshiro128p_next
  random_next_uniform_float32 = cuda.random.xoroshiro128p_uniform_float32

# %% [markdown]
# ## Compute capabilities

# %%
print(f'The number of CPU threads is {multiprocessing.cpu_count()}.')

# %%
MULTIPROCESS_IS_AVAILABLE = 'fork' in multiprocessing.get_all_start_methods()
print(f'Support for multiprocess "fork": {MULTIPROCESS_IS_AVAILABLE}')

# %%
if cuda.is_available() and cuda.detect():
  print(f'The number of GPU SMs is {cuda.get_current_device().MULTIPROCESSOR_COUNT}')


# %% [markdown]
# ## Shared code


# %%
DECK_SIZE = 52
NUM_SUITS = 4
NUM_RANKS = 13
HAND_SIZE = 5
HANDS_PER_DECK = DECK_SIZE - HAND_SIZE + 1  # We consider all overlapping hands in a shuffled deck.
CARDS_FOR_A_FLUSH = 5
CARDS_FOR_A_STRAIGHT = 5
RNG = np.random.default_rng(1)


# %%
class Outcome(enum.IntEnum):
  """Poker hand rankings from best to worst."""

  string_name: str
  reference_count: int  # https://en.wikipedia.org/wiki/Poker_probability

  ROYAL_FLUSH = 0, 'Royal flush', comb(4, 1)
  STRAIGHT_FLUSH = 1, 'Straight flush', comb(10, 1) * comb(4, 1) - comb(4, 1)
  FOUR_OF_A_KIND = 2, 'Four of a kind', comb(13, 1) * comb(4, 4) * comb(12, 1) * comb(4, 1)
  FULL_HOUSE = 3, 'Full house', comb(13, 1) * comb(4, 3) * comb(12, 1) * comb(4, 2)
  FLUSH = 4, 'Flush', comb(13, 5) * comb(4, 1) - comb(10, 1) * comb(4, 1)
  STRAIGHT = 5, 'Straight', comb(10, 1) * comb(4, 1) ** 5 - comb(10, 1) * comb(4, 1)
  THREE_OF_A_KIND = 6, 'Three of a kind', comb(13, 1) * comb(4, 3) * comb(12, 2) * comb(4, 1) ** 2
  TWO_PAIR = 7, 'Two pair', comb(13, 2) * comb(4, 2) ** 2 * comb(11, 1) * comb(4, 1)
  ONE_PAIR = 8, 'One pair', comb(13, 1) * comb(4, 2) * comb(12, 3) * comb(4, 1) ** 3
  HIGH_CARD = 9, 'High card', (comb(13, 5) - comb(10, 1)) * (comb(4, 1) ** 5 - comb(4, 1))

  def __new__(cls, value: int, string_name: str, reference_count: int) -> Any:
    obj = int.__new__(cls, value)
    obj._value_ = value
    obj.string_name = string_name
    obj.reference_count = reference_count
    return obj


# %%
NUM_OUTCOMES = len(Outcome)
EXPECTED_PROB = np.array([o.reference_count for o in Outcome]) / comb(DECK_SIZE, HAND_SIZE)
assert np.allclose(np.sum(EXPECTED_PROB), 1.0)


# %%
def write_cuda_kernel_assembly_code(function, filename):
  (ptx,) = function.inspect_asm().values()
  pathlib.Path(filename).write_text(ptx, encoding='utf-8')


# %%
def report_cuda_kernel_properties(function):
  PROPERTIES = 'const_mem_size local_mem_per_thread max_threads_per_block regs_per_thread shared_mem_per_block'.split()
  for property_name in PROPERTIES:
    (value,) = getattr(function, 'get_' + property_name)().values()
    print(f'{property_name} = {value}')


# %% [markdown]
# ## Approach 1: arrays

# %% [markdown]
# ### Hand evaluation


# %%
def outcome_of_hand_array_python(cards, ranks, freqs):
  """Evaluate 5-card poker hand and return outcome ranking, using array data structures.

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
outcome_of_hand_array_numba = numba.njit(outcome_of_hand_array_python)

# %% [markdown]
# ### CPU simulation


# %%
def make_compute_cpu(outcome_of_hand_array):
  # Return a specialized compute_cpu function for the given outcome_of_hand_array function.

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
          outcome = outcome_of_hand_array(hand, ranks, freqs)
          tally[outcome] += 1
    return tally / (num_decks * HANDS_PER_DECK)

  return compute_cpu


# %%
simulate_hands_array_cpu_python = make_compute_cpu(outcome_of_hand_array_python)
simulate_hands_array_cpu_numba = numba.njit(make_compute_cpu(outcome_of_hand_array_numba))


# %%
assert np.allclose(simulate_hands_array_cpu_numba(10**5, RNG), EXPECTED_PROB, atol=0.001)


# %%
# %timeit -n1 -r2 simulate_hands_array_cpu_python(2000, RNG)  # ~710 ms.

# %%
# %timeit -n100 -r2 simulate_hands_array_cpu_numba(10**4, RNG)  # ~14.8 ms.


# %%
def compute_chunk(args):
  num_decks, rng = args
  return simulate_hands_array_cpu_numba(num_decks, rng)


# %%
def simulate_hands_array_cpu_numba_multiprocess(num_decks, rng):
  num_processes = multiprocessing.cpu_count()
  chunk_num_decks = math.ceil(num_decks / num_processes)
  chunks = [(chunk_num_decks, np.random.default_rng(rng)) for i in range(num_processes)]
  with multiprocessing.get_context('fork').Pool(num_processes) as pool:
    results = pool.map(compute_chunk, chunks)
  return np.mean(results, axis=0)


# %% [markdown]
# ### GPU simulation


# %%
THREADS_PER_BLOCK = 256


# %%
@cuda.jit  # Using either local memory or cuda.shared.array.
def gpu_array(rng_states, num_decks_per_thread, global_tally):
  # pylint: disable=too-many-function-args, no-value-for-parameter, comparison-with-callable
  # pylint: disable=possibly-used-before-assignment
  USE_SHARED_MEMORY = True  # Shared memory is faster.
  thread_index = cuda.grid(1)
  if thread_index >= len(rng_states):
    return

  thread_id = cuda.threadIdx.x  # Index within block.
  if USE_SHARED_MEMORY:
    block_tally = cuda.shared.array((THREADS_PER_BLOCK, NUM_OUTCOMES), np.int32)
    block_deck = cuda.shared.array((THREADS_PER_BLOCK, DECK_SIZE), np.uint8)
    block_ranks = cuda.shared.array((THREADS_PER_BLOCK, HAND_SIZE), np.uint8)
    block_freqs = cuda.shared.array((THREADS_PER_BLOCK, NUM_RANKS), np.uint8)
    tally = block_tally[thread_id]
    deck = block_deck[thread_id]
    ranks = block_ranks[thread_id]
    freqs = block_freqs[thread_id]
    # Note: transposed structure results in longer code and ~1.3x slower execution, e.g.:
    # block_tally = cuda.shared.array((NUM_OUTCOMES, THREADS_PER_BLOCK), np.int32)
    # tally = block_tally[:, thread_id]
  else:
    tally = cuda.local.array(NUM_OUTCOMES, np.int32)
    deck = cuda.local.array(DECK_SIZE, np.uint8)
    ranks = cuda.local.array(HAND_SIZE, np.uint8)
    freqs = cuda.local.array(NUM_RANKS, np.uint8)

  tally[:] = 0
  for i in range(numba.uint8(DECK_SIZE)):  # Casting as uint8 nicely unrolls the loop.
    deck[i] = i

  for _ in range(num_decks_per_thread):
    # Apply Fisher-Yates shuffle to current deck.
    for i in range(51, 0, -1):
      random_uint32 = random_next_uniform_uint(rng_states, thread_index)
      j = random_uint32 % numba.uint32(i + 1)
      deck[i], deck[j] = deck[j], deck[i]

    for hand_index in range(HANDS_PER_DECK):
      hand = deck[hand_index : hand_index + 5]
      outcome = outcome_of_hand_array_numba(hand, ranks, freqs)
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
def simulate_hands_array_gpu_cuda(num_decks, rng):
  device = cuda.get_current_device()
  # Target enough threads for ~4 blocks per SM.
  target_num_threads = 4 * device.MULTIPROCESSOR_COUNT * THREADS_PER_BLOCK
  num_decks_per_thread = max(1, num_decks // target_num_threads)
  num_threads = num_decks // num_decks_per_thread
  # print(f'{num_decks_per_thread=} {num_threads=}')
  blocks = math.ceil(num_threads / THREADS_PER_BLOCK)
  seed = rng.integers(2**64, dtype=np.uint64)
  d_rng_states = random_create_states(num_threads, seed)
  d_global_tally = cuda.to_device(np.zeros(NUM_OUTCOMES, np.int64))
  gpu_array[blocks, THREADS_PER_BLOCK](d_rng_states, num_decks_per_thread, d_global_tally)
  return d_global_tally.copy_to_host() / (num_threads * num_decks_per_thread * HANDS_PER_DECK)


# %%
assert np.allclose(simulate_hands_array_gpu_cuda(10**7, RNG), EXPECTED_PROB, atol=0.0001)

# %%
# %timeit -n1 -r5 simulate_hands_array_gpu_cuda(10**7, RNG)  # ~220 ms.

# %%
if cuda.is_available():
  write_cuda_kernel_assembly_code(gpu_array, 'gpu_array.ptx')
  report_cuda_kernel_properties(gpu_array)

# %% [markdown]
# ## Approach 2: bitmasks


# %%
# Thanks to Marcel Gavriliu for this approach of storing card bitmasks and summing them.

# %%
CARD_COUNT_BITS = 3  # We use 3 bits to encode 0-7 cards for both ranks and suits.
RANKS_ONE = 0b_001_001_001_001_001_001_001_001_001_001_001_001_001
RANKS_TWO = RANKS_ONE << 1
RANKS_FOUR = RANKS_ONE << 2
ROYAL_STRAIGHT_RANK_MASK = 0b_001_001_001_001_001_000_000_000_000_000_000_000_000
SUITS_ONE = 0b_001_001_001_001

# %%
assert RANKS_ONE == int(sum((2**CARD_COUNT_BITS) ** np.arange(NUM_RANKS)))
assert SUITS_ONE == int(sum((2**CARD_COUNT_BITS) ** np.arange(NUM_SUITS)))
assert ROYAL_STRAIGHT_RANK_MASK == int(
    sum(8 ** np.arange(NUM_RANKS - CARDS_FOR_A_STRAIGHT, NUM_RANKS))
)


# %%
def create_table_mask_of_card():
  lst = [
      (1 << (CARD_COUNT_BITS * (NUM_RANKS + suit))) | (1 << (CARD_COUNT_BITS * rank))
      for suit in range(NUM_SUITS)
      for rank in range(NUM_RANKS)
  ]
  return np.array(lst, np.uint64)


# %%
# Table of 52 uint64, each the "(4 + 13) * 3"-bit mask encoding of the suit and rank of a card.
TABLE_MASK_OF_CARD = create_table_mask_of_card()


# %%
def create_table_straights_rank_mask():
  lst = [0b_001_001_001_001_001 << (i * 3) for i in range(9)]  # '23456' to 'TJQKA'.
  lst.append(0b_001_000_000_000_000_000_000_000_000_001_001_001_001)  # '2345A'.
  return np.array(lst, np.uint64)


# %%
# Table of 10 uint64, each the "13 * 3"-bit mask encoding of ranks in straight.
TABLE_STRAIGHTS_RANK_MASK = create_table_straights_rank_mask()


# %%
@numba.jit
def outcome_of_hand_bitmask(bitmask_sum):
  """Evaluate 5-card poker hand and return outcome ranking, using sum of card bitmasks."""
  # pylint: disable=too-many-function-args
  suit_count_mask = numba.uint32(bitmask_sum >> CARD_COUNT_BITS * NUM_RANKS)
  rank_count_mask = numba.uint64(bitmask_sum & (2 ** (CARD_COUNT_BITS * NUM_RANKS) - 1))

  is_flush = cuda.popc(numba.uint32(suit_count_mask & (suit_count_mask >> 2) & SUITS_ONE)) != 0

  # is_straight = rank_count_mask in TABLE_STRAIGHTS_RANK_MASK
  is_straight = False
  for straight_mask in TABLE_STRAIGHTS_RANK_MASK:  # Automatic cuda.const.array_like().
    if rank_count_mask == straight_mask:
      is_straight = True

  is_four = (rank_count_mask & RANKS_FOUR) != 0
  is_three = ((rank_count_mask + RANKS_ONE) & RANKS_FOUR) != 0
  mask_two_or_more = rank_count_mask & RANKS_TWO
  num_two_or_more = numba.uint32(cuda.popc(mask_two_or_more))  # Count number of set bits.

  if is_flush and is_straight:
    if (rank_count_mask & ROYAL_STRAIGHT_RANK_MASK) != 0:
      return Outcome.ROYAL_FLUSH.value
    return Outcome.STRAIGHT_FLUSH.value
  if is_four:
    return Outcome.FOUR_OF_A_KIND.value
  if is_three and num_two_or_more > 1:
    return Outcome.FULL_HOUSE.value
  if is_flush:
    return Outcome.FLUSH.value
  if is_straight:
    return Outcome.STRAIGHT.value
  if is_three:
    return Outcome.THREE_OF_A_KIND.value
  if num_two_or_more == 2:
    return Outcome.TWO_PAIR.value
  if num_two_or_more == 1:
    return Outcome.ONE_PAIR.value
  return Outcome.HIGH_CARD.value


# %%
@cuda.jit
def gpu_bitmask(rng_states, num_decks_per_thread, global_tally):
  # pylint: disable=too-many-function-args, no-value-for-parameter, comparison-with-callable
  thread_index = cuda.grid(1)
  if thread_index >= len(rng_states):
    return

  thread_id = cuda.threadIdx.x  # Index within block.
  block_tally = cuda.shared.array((THREADS_PER_BLOCK, NUM_OUTCOMES), np.int32)
  block_deck = cuda.shared.array((THREADS_PER_BLOCK, DECK_SIZE), np.uint8)
  tally = block_tally[thread_id]
  deck = block_deck[thread_id]

  tally[:] = 0
  for i in range(numba.uint8(DECK_SIZE)):  # Casting as uint8 nicely unrolls the loop.
    deck[i] = i

  for _ in range(num_decks_per_thread):
    # Apply Fisher-Yates shuffle to current deck.
    for i in range(51, 0, -1):
      random_uint32 = random_next_uniform_uint(rng_states, thread_index)
      j = random_uint32 % numba.uint32(i + 1)
      deck[i], deck[j] = deck[j], deck[i]

    MASK = cuda.const.array_like(TABLE_MASK_OF_CARD)
    mask0, mask1, mask2, mask3 = MASK[deck[0]], MASK[deck[1]], MASK[deck[2]], MASK[deck[3]]
    bitmask_sum = mask0 + mask1 + mask2 + mask3

    for hand_index in range(HANDS_PER_DECK):
      mask4 = MASK[deck[hand_index + 4]]
      bitmask_sum += mask4
      outcome = outcome_of_hand_bitmask(bitmask_sum)
      tally[outcome] += 1
      bitmask_sum -= mask0
      mask0, mask1, mask2, mask3 = mask1, mask2, mask3, mask4

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
def simulate_hands_mask_gpu_cuda(num_decks, rng):
  device = cuda.get_current_device()
  # Target enough threads for ~4 blocks per SM.
  target_num_threads = 4 * device.MULTIPROCESSOR_COUNT * THREADS_PER_BLOCK
  num_decks_per_thread = max(1, num_decks // target_num_threads)
  num_threads = num_decks // num_decks_per_thread
  # print(f'{num_decks_per_thread=} {num_threads=}')
  blocks = math.ceil(num_threads / THREADS_PER_BLOCK)
  seed = rng.integers(2**64, dtype=np.uint64)
  d_rng_states = random_create_states(num_threads, seed)
  d_global_tally = cuda.to_device(np.zeros(NUM_OUTCOMES, np.int64))
  gpu_bitmask[blocks, THREADS_PER_BLOCK](d_rng_states, num_decks_per_thread, d_global_tally)
  return d_global_tally.copy_to_host() / (num_threads * num_decks_per_thread * HANDS_PER_DECK)


# %%
assert np.allclose(simulate_hands_mask_gpu_cuda(10**7, RNG), EXPECTED_PROB, atol=0.0001)

# %%
# %timeit -n1 -r5 simulate_hands_mask_gpu_cuda(10**7, RNG)  # ~50-100 ms.

# %%
if cuda.is_available():
  write_cuda_kernel_assembly_code(gpu_bitmask, 'gpu_bitmask.ptx')
  report_cuda_kernel_properties(gpu_bitmask)

# %% [markdown]
# ## Results

# %%
SIMULATE_FUNCTIONS = {
    name: func
    for name, func in {
        'array_cpu_python': simulate_hands_array_cpu_python,
        'array_cpu_numba': simulate_hands_array_cpu_numba,
        'array_cpu_numba_multiprocess': simulate_hands_array_cpu_numba_multiprocess,
        'array_gpu_cuda': simulate_hands_array_gpu_cuda,
        'mask_gpu_cuda': simulate_hands_mask_gpu_cuda,
    }.items()
    if ('multiprocess' not in name or MULTIPROCESS_IS_AVAILABLE)
    and ('cuda' not in name or cuda.is_available())
}

# %%
COMPLEXITY_ADJUSTMENT = {
    'array_cpu_python': 0.01,
    'array_cpu_numba': 2.0,
    'array_cpu_numba_multiprocess': 20.0,
    'array_gpu_cuda': 100.0,
    'mask_gpu_cuda': 100.0,
}


# %%
def simulate_poker_hands(base_num_hands, func_name, func):
  base_num_decks = math.ceil(base_num_hands / HANDS_PER_DECK)
  num_decks = math.ceil(base_num_decks * COMPLEXITY_ADJUSTMENT[func_name])
  num_hands = num_decks * HANDS_PER_DECK
  print(f'\nFor {func_name} simulating {num_hands:,} hands:')

  # Ensure the function is jitted.
  _ = func(int(100_000 * COMPLEXITY_ADJUSTMENT[func_name]), RNG)

  start_time = time.perf_counter_ns()
  results = func(num_decks, RNG)
  elapsed_time = (time.perf_counter_ns() - start_time) / 10**9

  round_digits = lambda x, ndigits=3: round(x, ndigits - 1 - math.floor(math.log10(abs(x))))
  hands_per_s = round_digits(int(num_hands / elapsed_time))
  print(f' Elapsed time is {elapsed_time:.3f} s, or {hands_per_s:,} hands/s.')

  print(' Probabilities:')
  for outcome, result_prob in zip(Outcome, results):
    reference_prob = outcome.reference_count / comb(DECK_SIZE, HAND_SIZE)
    error = result_prob - reference_prob
    s = f'  {outcome.string_name:<16}: {result_prob * 100:8.5f}%'
    s += f'  (vs. reference {reference_prob * 100:8.5f}%  error:{error * 100:8.5f}%)'
    print(s)


# %%
def compare_simulations(base_num_hands):
  for func_name, func in SIMULATE_FUNCTIONS.items():
    simulate_poker_hands(base_num_hands, func_name, func)


# %%
compare_simulations(base_num_hands=10**7)

# %%
# 135k, 33m, 350m, 2200-3400m, 6000m-7200m

# %% [markdown]
# ## End
