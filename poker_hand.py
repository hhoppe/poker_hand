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
# - [Open in kaggle.com](https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/poker_hand/blob/main/poker_hand.ipynb), login, Session options -> Accelerator -> GPU T4 x2 or P100, Session options -> Internet -> On, then Run All.
#
# - [Open in mybinder.org](https://mybinder.org/v2/gh/hhoppe/poker_hand/main?urlpath=lab/tree/poker_hand.ipynb).  Unfortunately, no GPU is available.
#
# - [Open in deepnote.com](https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fpoker_hand%2Fblob%2Fmain%2Fpoker_hand.ipynb).  Unfortunately, no GPU is available.
#
# Here are results:

# %% [markdown]
# <table style="margin-left: 0">
# <tr>
#   <th colspan="4"></th>
#   <th colspan="8" style="text-align: center; background-color: #F4F4F4">Simulation rates (hands/s)</th>
# </tr>
# <tr>
#   <th colspan="4">Compute capabilities</th>
#   <th colspan="4" style="text-align: center; background-color: #EBF5FF">Array-based</th>
#   <th colspan="3" style="text-align: center; background-color: #F0FDF4">Bitcount-based</th>
#   <th colspan="1" style="text-align: center; background-color: #FFF8F4">Deckmask-based</th>
# </tr>
# <tr>
#   <th>Platform</th>
#   <th style="text-align: center">CPU<br>threads</th>
#   <th style="text-align: center">GPU<br>type</th>
#   <th style="text-align: center">CUDA<br>SMs</th>
#   <th style="background-color: #EBF5FF">Python</th>
#   <th style="background-color: #EBF5FF">Numba</th>
#   <th style="background-color: #EBF5FF">Multiprocess</th>
#   <th style="background-color: #EBF5FF">CUDA</th>
#   <th style="background-color: #F0FDF4">Numba</th>
#   <th style="background-color: #F0FDF4">Multiprocess</th>
#   <th style="background-color: #F0FDF4">CUDA</th>
#   <th style="background-color: #FFF8F4">CUDA</th>
# </tr>
# <tr>
#   <td><b>My PC</b> Win10</td>
#   <td style="text-align: center">24</td>
#   <td style="text-align: center">GeForce 3080 Ti</td>
#   <td style="text-align: center">80</td>
#   <td style="text-align: right; background-color: #EBF5FF">21,800</td>
#   <td style="text-align: right; background-color: #EBF5FF">28,200,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">-</td>
#   <td style="text-align: right; background-color: #EBF5FF">3,300,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">125,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">-</td>
#   <td style="text-align: right; background-color: #F0FDF4">12,200,000,000</td>
#   <td style="text-align: right; background-color: #FFF8F4">?,200,000,000</td>
# </tr>
# <tr>
#   <td><b>My PC</b> WSL2</td>
#   <td style="text-align: center">24</td>
#   <td style="text-align: center">GeForce 3080 Ti</td>
#   <td style="text-align: center">80</td>
#   <td style="text-align: right; background-color: #EBF5FF">135,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">33,000,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">350,000,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">16,500,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">138,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">912,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">70,000,000,000</td>
#   <td style="text-align: right; background-color: #FFF8F4">148,000,000,000</td>
# </tr>
# <tr>
#   <td><b>Marcel PC</b> Win</td>
#   <td style="text-align: center">40</td>
#   <td style="text-align: center">Titan V</td>
#   <td style="text-align: center">80</td>
#   <td style="text-align: right; background-color: #EBF5FF"><s>62,800</s></td>
#   <td style="text-align: right; background-color: #EBF5FF"><s>5,160,000</s></td>
#   <td style="text-align: right; background-color: #EBF5FF">-</td>
#   <td style="text-align: right; background-color: #EBF5FF"><s>2,280,000,000</s></td>
#   <td style="text-align: right; background-color: #F0FDF4">?00,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">-</td>
#   <td style="text-align: right; background-color: #F0FDF4">?,000,000,000</td>
#   <td style="text-align: right; background-color: #FFF8F4">?,000,000,000</td>
# </tr>
# <tr>
#   <td><a href="https://colab.research.google.com/github/hhoppe/poker_hand/blob/main/poker_hand.ipynb"><b>Colab</b> T4</a></td>
#   <td style="text-align: center">2</td>
#   <td style="text-align: center">Tesla T4</td>
#   <td style="text-align: center">40</td>
#   <td style="text-align: right; background-color: #EBF5FF">17,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">14,600,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">16,400,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">3,670,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">60,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">63,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">32,200,000,000</td>
#   <td style="text-align: right; background-color: #FFF8F4">32,200,000,000</td>
# </tr>
# <tr>
#   <td><a href="https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/poker_hand/blob/main/poker_hand.ipynb"><b>Kaggle</b> T4</a></td>
#   <td style="text-align: center">4</td>
#   <td style="text-align: center">Tesla T4 x2</td>
#   <td style="text-align: center">40</td>
#   <td style="text-align: right; background-color: #EBF5FF">17,300</td>
#   <td style="text-align: right; background-color: #EBF5FF">14,800,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">31,800,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">3,560,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">61,600,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">117,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">31,800,000,000</td>
#   <td style="text-align: right; background-color: #FFF8F4">31,800,000,000</td>
# </tr>
# <tr>
#   <td><a href="https://www.kaggle.com/notebooks/welcome?src=https://github.com/hhoppe/poker_hand/blob/main/poker_hand.ipynb"><b>Kaggle</b> P100</a></td>
#   <td style="text-align: center">4</td>
#   <td style="text-align: center">Tesla P100</td>
#   <td style="text-align: center">56</td>
#   <td style="text-align: right; background-color: #EBF5FF">17,100</td>
#   <td style="text-align: right; background-color: #EBF5FF">15,000,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">31,000,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">3,600,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">62,400,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">124,000,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">34,700,000,000</td>
#   <td style="text-align: right; background-color: #FFF8F4">34,700,000,000</td>
# </tr>
# <tr>
#   <td><a href="https://mybinder.org/v2/gh/hhoppe/poker_hand/main?urlpath=lab/tree/poker_hand.ipynb"><b>Mybinder</b></a></td>
#   <td style="text-align: center">72</td>
#   <td style="text-align: center">None</td>
#   <td style="text-align: center">-</td>
#   <td style="text-align: right; background-color: #EBF5FF">18,200</td>
#   <td style="text-align: right; background-color: #EBF5FF">7,100,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">3,200,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">-</td>
#   <td style="text-align: right; background-color: #F0FDF4">23,500,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">9,600,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">-</td>
#   <td style="text-align: right; background-color: #FFF8F4">-</td>
# </tr>
# <tr>
#   <td><a href="https://deepnote.com/launch?url=https%3A%2F%2Fgithub.com%2Fhhoppe%2Fpoker_hand%2Fblob%2Fmain%2Fpoker_hand.ipynb"><b>Deepnote</b></a></td>
#   <td style="text-align: center">8</td>
#   <td style="text-align: center">None</td>
#   <td style="text-align: center">-</td>
#   <td style="text-align: right; background-color: #EBF5FF">16,600</td>
#   <td style="text-align: right; background-color: #EBF5FF">23,700,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">12,900,000</td>
#   <td style="text-align: right; background-color: #EBF5FF">-</td>
#   <td style="text-align: right; background-color: #F0FDF4">72,300,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">47,500,000</td>
#   <td style="text-align: right; background-color: #F0FDF4">-</td>
#   <td style="text-align: right; background-color: #FFF8F4">-</td>
# </tr>
# </table>

# %% [markdown]
# It is puzzling that the CUDA rate is lower on my PC than on the online servers.  The GeForce may have worse uint64 throughput?

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
import typing
from typing import Any, Callable

import numba
from numba import cuda
import numba.cuda.random
import numpy as np
import numpy.typing
import random32  # Replacement for https://github.com/numba/numba/blob/main/numba/cuda/random.py

# %%
# mypy: disable-error-code="no-any-return"

# %%
_NDArray = numpy.typing.NDArray[Any]
_CudaArray = Any  # cuda.cudadrv.devicearray.DeviceNDArray
_F = typing.TypeVar('_F', bound=Callable[..., Any])

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
Probabilities = _NDArray  # Or np.ndarray[tuple[Literal[10]], np.dtype[np.float64]]
EXPECTED_PROB = np.array([o.reference_count for o in Outcome]) / comb(DECK_SIZE, HAND_SIZE)
assert np.allclose(np.sum(EXPECTED_PROB), 1.0)


# %%
def numba_njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
  """Typed replacement for non-bare `@numba.njit()` decorator."""
  assert kwargs or not (len(args) == 1 and callable(args[0]))

  def decorator(func: _F) -> _F:
    jitted_func: _F = numba.njit(*args, **kwargs)(func)
    return jitted_func

  return decorator


# %%
def popc_helper(typing_context: Any, src: Any) -> Any:
  """Helper function to enable the intrinsic CPU instruction `popc()`."""
  del typing_context
  if not isinstance(src, numba.types.Integer):
    return None
  DICT = {numba.uint64: 'i64', numba.uint32: 'i32', numba.uint16: 'i16', numba.uint8: 'i8'}
  DICT |= {numba.int64: 'i64', numba.int32: 'i32', numba.int16: 'i16', numba.int8: 'i8'}
  llvm_type = DICT[src]

  def codegen(context: Any, builder: Any, signature: Any, args: Any) -> Any:
    del context, signature
    return numba.cpython.mathimpl.call_fp_intrinsic(builder, 'llvm.ctpop.' + llvm_type, args)

  return src(src), codegen


popc_intrinsic: Callable[[int], int] = numba.extending.intrinsic(popc_helper)


@numba_njit(inline='always')
def cpu_popc(x: int) -> int:  # https://stackoverflow.com/a/77103233
  """Return the ("population") count of set bits in an integer, using a CPU intrinsic."""
  return popc_intrinsic(x)


# %%
@numba_njit(inline='always')
def gpu_popc(x: int) -> int:
  """Return the ("population") count of set bits in an integer, using a CUDA intrinsic."""
  return cuda.popc(x)  # pylint: disable=too-many-function-args


# %%
def write_numba_assembly_code(function: Callable[..., Any], filename: str) -> None:
  """Write the asm or ptx code for a numba function to an output file."""
  (asm_or_ptx,) = function.inspect_asm().values()  # type: ignore[attr-defined]
  pathlib.Path(filename).write_text(asm_or_ptx, encoding='utf-8')


# %%
def report_cuda_kernel_properties(function: Callable[..., Any]) -> None:
  """Show memory attributes of a CUDA kernel function."""
  PROPERTIES = 'const_mem_size local_mem_per_thread max_threads_per_block regs_per_thread shared_mem_per_block'.split()
  for property_name in PROPERTIES:
    (value,) = getattr(function, 'get_' + property_name)().values()
    print(f'{property_name} = {value}')


# %% [markdown]
# ## Approach 1: card arrays

# %% [markdown]
# We represent a card as a 6-bit integer in `range(52)`, with the suit `range(4)` in the 2 lsb and the rank `range(13)` in the top 4 msb.
#
# A hand is a tuple of five such integers.

# %% [markdown]
# ### Hand evaluation


# %%
def outcome_of_hand_array_python(cards: Any, ranks: Any, freqs: Any) -> Outcome:
  """Evaluate a 5-card poker hand and return its outcome ranking, using array data structures.

  Args:
    cards: Array of 5 integers representing cards (0-51).
    ranks: Pre-allocated array for storing sorted ranks.
    freqs: Pre-allocated array for counting rank frequencies.

  Returns:
    An enum representing the hand ranking.
  """
  # pylint: disable=consider-using-in

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
    max_freq = max(max_freq, freq)
    if freq == 2:
      num_pairs += 1

  if is_flush and is_straight:
    if ranks[0] == 8:
      return Outcome.ROYAL_FLUSH
    return Outcome.STRAIGHT_FLUSH
  if max_freq == 4:
    return Outcome.FOUR_OF_A_KIND
  if max_freq == 3 and num_pairs > 0:
    return Outcome.FULL_HOUSE
  if is_flush:
    return Outcome.FLUSH
  if is_straight:
    return Outcome.STRAIGHT
  if max_freq == 3:
    return Outcome.THREE_OF_A_KIND
  if max_freq == 2:
    if num_pairs == 2:
      return Outcome.TWO_PAIR
    return Outcome.ONE_PAIR
  return Outcome.HIGH_CARD


# %%
outcome_of_hand_array_numba = numba_njit(inline='always')(outcome_of_hand_array_python)

# %% [markdown]
# ### CPU simulation


# %%
def make_compute_array_cpu(for_numba: bool) -> Callable[[int, np.random.Generator], Probabilities]:
  """Factory returning a function for Python or numba evaluation."""
  outcome_of_hand_array = outcome_of_hand_array_numba if for_numba else outcome_of_hand_array_python

  def compute_array_cpu(num_decks: int, rng: np.random.Generator) -> Probabilities:
    """Return hand probabilities computed on the CPU using array data structures."""
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
          tally[outcome.value] += 1
    return tally / (num_decks * HANDS_PER_DECK)

  return compute_array_cpu


# %%
simulate_hands_array_cpu_python = make_compute_array_cpu(for_numba=False)
simulate_hands_array_cpu_numba = numba_njit()(make_compute_array_cpu(for_numba=True))


# %%
if 0:
  write_numba_assembly_code(simulate_hands_array_cpu_numba, 'simulate_hands_array_cpu_numba.asm')

# %%
assert np.allclose(simulate_hands_array_cpu_numba(10**5, RNG), EXPECTED_PROB, atol=0.001)


# %%
# %timeit -n1 -r2 simulate_hands_array_cpu_python(2000, RNG)  # ~750 ms.

# %%
# %timeit -n100 -r2 simulate_hands_array_cpu_numba(10**4, RNG)  # ~10.3 ms.


# %%
def compute_array_chunk(args: tuple[int, np.random.Generator]) -> Probabilities:
  """Helper function for multiprocess dispatching."""
  num_decks, rng = args
  return simulate_hands_array_cpu_numba(num_decks, rng)


# %%
def simulate_hands_array_cpu_numba_multiprocess(
    num_decks: int, rng: np.random.Generator
) -> Probabilities:
  """Compute hand probabilities using multiprocessing."""
  num_processes = multiprocessing.cpu_count()
  chunk_num_decks = math.ceil(num_decks / num_processes)
  new_rngs = rng.spawn(num_processes)
  chunks = [(chunk_num_decks, new_rng) for new_rng in new_rngs]
  with multiprocessing.get_context('fork').Pool(num_processes) as pool:
    results = pool.map(compute_array_chunk, chunks)
  return np.mean(results, axis=0)


# %% [markdown]
# ### GPU simulation


# %%
THREADS_PER_BLOCK = 256


# %%
# Using either local memory or cuda.shared.array.
@cuda.jit  # type: ignore[misc]
def gpu_array(rng_states: _CudaArray, decks_per_thread: int, global_tally: _CudaArray) -> None:
  """CUDA kernel to compute hand probabilities using an array approach."""
  # pylint: disable=too-many-function-args, no-value-for-parameter, comparison-with-callable
  # pylint: disable=possibly-used-before-assignment
  int32, uint32 = numba.int32, numba.uint32
  USE_SHARED_MEMORY = True  # Shared memory is faster.
  thread_index = cuda.grid(1)
  if thread_index >= len(rng_states):
    return

  thread_id = cuda.threadIdx.x  # Index within block.
  if USE_SHARED_MEMORY:
    block_tally = cuda.shared.array((NUM_OUTCOMES, THREADS_PER_BLOCK), np.int32)
    block_deck = cuda.shared.array((THREADS_PER_BLOCK, DECK_SIZE), np.uint8)
    block_ranks = cuda.shared.array((THREADS_PER_BLOCK, HAND_SIZE), np.uint8)
    block_freqs = cuda.shared.array((THREADS_PER_BLOCK, NUM_RANKS), np.uint8)
    tally = block_tally[:, thread_id]
    deck = block_deck[thread_id]
    ranks = block_ranks[thread_id]
    freqs = block_freqs[thread_id]
  else:
    tally = cuda.local.array(NUM_OUTCOMES, np.int32)
    deck = cuda.local.array(DECK_SIZE, np.uint8)
    ranks = cuda.local.array(HAND_SIZE, np.uint8)
    freqs = cuda.local.array(NUM_RANKS, np.uint8)

  tally[:] = 0
  for i in range(numba.uint8(DECK_SIZE)):  # Casting as uint8 nicely unrolls the loop.
    deck[i] = i

  for _ in range(int32(decks_per_thread)):
    # Apply Fisher-Yates shuffle to current deck.
    # for i in range(int32(51), int32(0), int32(-1)):
    i = uint32(51)
    while i > 0:
      random_uint = random32.xoshiro128p_next(rng_states, thread_index)
      j = random_uint % uint32(i + 1)
      deck[i], deck[j] = deck[j], deck[i]
      i = uint32(i - 1)

    # for hand_index in range(HANDS_PER_DECK):
    hand_index = uint32(0)
    while hand_index < HANDS_PER_DECK:
      hand = deck[hand_index : hand_index + 5]
      outcome = uint32(outcome_of_hand_array_numba(hand, ranks, freqs))
      tally[outcome] += 1
      hand_index = uint32(hand_index + 1)

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
def simulate_hands_array_gpu_cuda(num_decks: int, rng: np.random.Generator) -> Probabilities:
  """Compute hand probabilities using an array approach by invoking a CUDA kernel."""
  device = cuda.get_current_device()
  PARALLELISM = 8  # Number of blocks per SM.
  target_num_threads = PARALLELISM * device.MULTIPROCESSOR_COUNT * THREADS_PER_BLOCK
  decks_per_thread = max(1, num_decks // target_num_threads)
  num_threads = num_decks // decks_per_thread
  # print(f'{decks_per_thread=} {num_threads=}')
  blocks = math.ceil(num_threads / THREADS_PER_BLOCK)
  seed = rng.integers(2**64, dtype=np.uint64)
  d_rng_states = random32.create_xoshiro128p_states(num_threads, seed)
  d_global_tally = cuda.to_device(np.zeros(NUM_OUTCOMES, np.int64))
  gpu_array[blocks, THREADS_PER_BLOCK](d_rng_states, decks_per_thread, d_global_tally)
  return d_global_tally.copy_to_host() / (num_threads * decks_per_thread * HANDS_PER_DECK)


# %%
if cuda.is_available():
  assert np.allclose(simulate_hands_array_gpu_cuda(10**7, RNG), EXPECTED_PROB, atol=0.0001)

# %%
if cuda.is_available():
  print('Timing:')
  # %timeit -n1 -r5 simulate_hands_array_gpu_cuda(2 * 10**7, RNG)  # ~250-400 ms.

# %%
if cuda.is_available():
  write_numba_assembly_code(gpu_array, 'gpu_array.ptx')
  report_cuda_kernel_properties(gpu_array)

# %% [markdown]
# ## Approach 2: suit/rank bitcounts


# %% [markdown]
# We represent a (5-card) hand as a sequence of $4+13$ counts, each of size 3 bits.
# The 3-bit counts encode:
# - the number of cards in each of the 4 suits, and
# - the number of cards of each of the 13 ranks (2,3,4,5,6,7,8,9,T,J,Q,K,A).
#
# Therefore each card is encoded as a 51-bit mask in which exactly **two bits** are set to 1,
# in the corresponding suit and rank.
#
# The bitcounts for the five cards in a hand are combined using ordinary 64-bit addition,
# relying on the fact that **no carry bits** will jump across the 3-bit triples because $5 < 2^3$.

# %%
# Thanks to Marcel Gavriliu for this approach of storing card bitmasks and summing them.

# %%
THREADS_PER_BLOCK = 256

# %%
CARD_COUNT_BITS = 3  # We use 3 bits to encode 0-7 cards for both ranks and suits.
RANKS_ONE = 0b_001_001_001_001_001_001_001_001_001_001_001_001_001  # One card of each rank.
RANKS_TWO = RANKS_ONE << 1
RANKS_FOUR = RANKS_ONE << 2
ROYAL_STRAIGHT_RANK_MASK = 0b_001_001_001_001_001_000_000_000_000_000_000_000_000
ACE_LOW_STRAIGHT_RANK_MASK = 0b_001_000_000_000_000_000_000_000_000_001_001_001_001
SUITS_ONE = 0b_001_001_001_001  # One card of each suit.

# %%
assert RANKS_ONE == int(sum((2**CARD_COUNT_BITS) ** np.arange(NUM_RANKS, dtype=np.uint64)))
assert SUITS_ONE == int(sum((2**CARD_COUNT_BITS) ** np.arange(NUM_SUITS, dtype=np.uint64)))


# %%
def create_table_straights_rank_mask() -> _NDArray:
  """Create a table containing the 10 bitmasks corresponding to the 10 possible straights."""
  lst = [0b_001_001_001_001_001 << (i * 3) for i in range(9)]  # '23456' to 'TJQKA'.
  lst.append(ACE_LOW_STRAIGHT_RANK_MASK)  # 'A2345'.
  return np.array(lst, np.uint64)


# %%
# Table of 10 uint64, each the "13 * 3"-bit mask encoding of ranks in straight.
TABLE_STRAIGHTS_RANK_MASK = create_table_straights_rank_mask()


# %% [markdown]
# ### Hand evaluation


# %%
@numba_njit(inline='always')
def bitcount_of_card(card: numba.uint8) -> numba.uint64:
  """Return a bitmask of 4*3 + 13*3 bits encoding the suit and rank of 0 <= `card` < 52."""
  suit_index = card & 0b11
  rank_index = card >> 2
  mask = (1 << (suit_index * 3 + NUM_RANKS * 3)) | (1 << (rank_index * 3))
  return mask


# %%
@numba_njit(inline='always')
def determine_straight(rank_count_mask: numba.uint64) -> bool:
  """Return true if the rank bitmask corresponds to a straight."""
  m = rank_count_mask
  t = TABLE_STRAIGHTS_RANK_MASK
  assert len(t) == 10
  result = (m == t[0]) | (m == t[1]) | (m == t[2]) | (m == t[3]) | (m == t[4])
  result |= (m == t[5]) | (m == t[6]) | (m == t[7]) | (m == t[8]) | (m == t[9])
  return result


# %%
def make_outcome_of_hand_bitcount(for_cuda: bool) -> Callable[[numba.uint64], Outcome]:
  """Factory returning a function for numba or cuda evaluation."""
  popc = gpu_popc if for_cuda else cpu_popc

  def outcome_of_hand_bitcount(bitcount_sum: numba.uint64) -> Outcome:
    """Evaluate 5-card poker hand and return outcome ranking, using sum of card bitcounts."""
    # pylint: disable=too-many-function-args
    uint32, uint64 = numba.uint32, numba.uint64
    suit_count_mask = uint32(bitcount_sum >> CARD_COUNT_BITS * NUM_RANKS)
    rank_count_mask = uint64(bitcount_sum & (2 ** (CARD_COUNT_BITS * NUM_RANKS) - 1))

    is_flush = popc(uint32(suit_count_mask & (suit_count_mask >> 2) & SUITS_ONE)) != 0
    is_straight = determine_straight(rank_count_mask)
    is_four = (rank_count_mask & RANKS_FOUR) != 0
    is_three = ((rank_count_mask + RANKS_ONE) & RANKS_FOUR) != 0
    mask_two_or_more = rank_count_mask & RANKS_TWO
    num_two_or_more = popc(mask_two_or_more)

    if is_flush and is_straight:
      if rank_count_mask == ROYAL_STRAIGHT_RANK_MASK:
        return Outcome.ROYAL_FLUSH
      return Outcome.STRAIGHT_FLUSH
    if is_four:
      return Outcome.FOUR_OF_A_KIND
    if is_three and num_two_or_more > 1:
      return Outcome.FULL_HOUSE
    if is_flush:
      return Outcome.FLUSH
    if is_straight:
      return Outcome.STRAIGHT
    if is_three:
      return Outcome.THREE_OF_A_KIND
    if num_two_or_more == 2:
      return Outcome.TWO_PAIR
    if num_two_or_more == 1:
      return Outcome.ONE_PAIR
    return Outcome.HIGH_CARD

  return outcome_of_hand_bitcount


# %%
outcome_of_hand_bitcount_numba = numba_njit()(make_outcome_of_hand_bitcount(for_cuda=False))
outcome_of_hand_bitcount_cuda = numba_njit()(make_outcome_of_hand_bitcount(for_cuda=True))


# %% [markdown]
# ### CPU simulation


# %%
@numba_njit()
def simulate_hands_bitcount_cpu_numba(num_decks: int, rng: np.random.Generator) -> Probabilities:
  """Compute hand probabilities using a bitcount approach by invoking numba-jitted CPU code."""
  BLOCK_SIZE = 10
  deck_block = np.empty((BLOCK_SIZE, DECK_SIZE), dtype=np.uint8)
  deck_block[:] = np.arange(DECK_SIZE, dtype=np.uint8)
  tally = np.zeros(NUM_OUTCOMES, np.int64)

  for deck_index_start in range(0, num_decks, BLOCK_SIZE):
    deck_block2 = rng.permutation(deck_block, axis=1)

    for deck_index in range(deck_index_start, min(deck_index_start + BLOCK_SIZE, num_decks)):
      deck = deck_block2[deck_index - deck_index_start]
      mask0, mask1 = bitcount_of_card(deck[0]), bitcount_of_card(deck[1])
      mask2, mask3 = bitcount_of_card(deck[2]), bitcount_of_card(deck[3])
      bitcount_sum = mask0 + mask1 + mask2 + mask3

      for hand_index in range(HANDS_PER_DECK):
        mask4 = bitcount_of_card(deck[hand_index + 4])
        bitcount_sum += mask4
        outcome = outcome_of_hand_bitcount_numba(bitcount_sum)
        tally[outcome.value] += 1
        bitcount_sum -= mask0
        mask0, mask1, mask2, mask3 = mask1, mask2, mask3, mask4

  return tally / (num_decks * HANDS_PER_DECK)


# %%
# simulate_poker_hands(10**7, 'bitcount_cpu_numba', simulate_hands_bitcount_cpu_numba)  # ~130 m hands/s

# %%
assert np.allclose(simulate_hands_bitcount_cpu_numba(10**5, RNG), EXPECTED_PROB, atol=0.002)


# %%
# %timeit -n2 -r2 simulate_hands_bitcount_cpu_numba(10**5, RNG)  # ~35 ms if low variance.


# %%
def compute_bitcount_chunk(args: tuple[int, np.random.Generator]) -> Probabilities:
  """Helper function for multiprocessing."""
  num_decks, rng = args
  return simulate_hands_bitcount_cpu_numba(num_decks, rng)


# %%
def simulate_hands_bitcount_cpu_numba_multiprocess(
    num_decks: int, rng: np.random.Generator
) -> Probabilities:
  """Compute hand probabilities using a bitcount approach and multiprocessing."""
  num_processes = multiprocessing.cpu_count()
  chunk_num_decks = math.ceil(num_decks / num_processes)
  new_rngs = rng.spawn(num_processes)
  chunks = [(chunk_num_decks, new_rng) for new_rng in new_rngs]
  with multiprocessing.get_context('fork').Pool(num_processes) as pool:
    results = pool.map(compute_bitcount_chunk, chunks)
  return np.mean(results, axis=0)


# %% [markdown]
# ### GPU simulation


# %%
@cuda.jit(fastmath=True)  # type: ignore[misc]
def gpu_bitcount(rng_states: _CudaArray, decks_per_thread: int, global_tally: _CudaArray) -> None:
  """CUDA kernel to compute hand probabilities using a bitcount approach."""
  # pylint: disable=too-many-function-args, no-value-for-parameter, comparison-with-callable
  int32, uint32, float32 = numba.int32, numba.uint32, numba.float32
  thread_index = cuda.grid(1)
  thread_id = cuda.threadIdx.x  # Index within block.

  block_tally = cuda.shared.array((NUM_OUTCOMES, THREADS_PER_BLOCK), np.int32)
  block_deck = cuda.shared.array((THREADS_PER_BLOCK, DECK_SIZE), np.uint8)
  tally = block_tally[:, thread_id]
  deck = block_deck[thread_id]
  tally[:] = 0

  if thread_index >= len(rng_states):
    return  # This must come after tally[] is zeroed, to simplify reduction below.

  for i in range(numba.uint8(DECK_SIZE)):  # Casting as uint8 nicely unrolls the loop.
    deck[i] = i

  rng = rng_states[thread_index]
  s0, s1, s2, s3 = uint32(rng['s0']), uint32(rng['s1']), uint32(rng['s2']), uint32(rng['s3'])

  for _ in range(int32(decks_per_thread)):
    # Apply Fisher-Yates shuffle to current deck.
    # for i in range(int32(51), int32(0), int32(-1)):
    i = uint32(51)
    while i > 0:
      # Swap element at [i] with any element [0, ..., i].
      random_uint32, s0, s1, s2, s3 = random32.xoshiro128p_next_raw(s0, s1, s2, s3)
      random_uint32 = uint32(random_uint32)
      s0, s1, s2, s3 = uint32(s0), uint32(s1), uint32(s2), uint32(s3)
      if 0:
        j = random_uint32 % uint32(i + 1)  # Remainder is somewhat expensive in CUDA.
      elif 0:
        # In `random_uint32`, the msb have better randomness than the lsb, so float32 mul is better.
        value_in_unit_interval = random32.uint32_to_unit_float32(random_uint32)
        j = uint32(float32(value_in_unit_interval * float32(int32(i + 1))))
      else:
        j = uint32(float32(random_uint32) * float32((1.0 - 1e-7) / 2**32) * float32(int32(i + 1)))
      deck[i], deck[j] = deck[j], deck[i]
      i = uint32(i - 1)

    mask0, mask1 = bitcount_of_card(deck[0]), bitcount_of_card(deck[1])
    mask2, mask3 = bitcount_of_card(deck[2]), bitcount_of_card(deck[3])
    bitcount_sum = mask0 + mask1 + mask2 + mask3

    # for hand_index in range(int32(HANDS_PER_DECK)):
    card_index = uint32(4)
    while card_index < uint32(DECK_SIZE):
      # mask4 = bitcount_of_card(deck[hand_index + 4])
      mask4 = bitcount_of_card(deck[card_index])
      bitcount_sum += mask4
      outcome = uint32(outcome_of_hand_bitcount_cuda(bitcount_sum).value)
      tally[outcome] += 1
      bitcount_sum -= mask0
      mask0, mask1, mask2, mask3 = mask1, mask2, mask3, mask4
      card_index = uint32(card_index + 1)

  # Compute a parallel sum reduction on the outcome tally.
  temp_tally = cuda.shared.array(NUM_OUTCOMES, np.int64)  # Per-block tally.
  temp_tally[:] = 0
  cuda.syncthreads()

  # First do a sum reduction within the 32 lanes of each warp (still at 32-bit precision).
  for i in range(NUM_OUTCOMES):
    value = tally[i]
    for offset in [16, 8, 4, 2, 1]:
      value += cuda.shfl_down_sync(0xFFFFFFFF, value, offset)
    if cuda.laneid == 0:
      cuda.atomic.add(temp_tally, i, numba.int64(value))  # Reduce across the block's warps.
  cuda.syncthreads()

  # Final reduction across blocks to global_tally.
  if thread_id == 0:
    for i in range(NUM_OUTCOMES):
      cuda.atomic.add(global_tally, i, temp_tally[i])


# %%
def simulate_hands_bitcount_gpu_cuda(num_decks: int, rng: np.random.Generator) -> Probabilities:
  """Compute hand probabilities using a bitcount approach by invoking a CUDA kernel."""
  device = cuda.get_current_device()
  PARALLELISM = 8  # Number of blocks per SM.
  target_num_threads = PARALLELISM * device.MULTIPROCESSOR_COUNT * THREADS_PER_BLOCK
  decks_per_thread = max(1, num_decks // target_num_threads)
  num_threads = num_decks // decks_per_thread
  # print(f'{decks_per_thread=} {num_threads=}')
  blocks = math.ceil(num_threads / THREADS_PER_BLOCK)
  seed = rng.integers(2**64, dtype=np.uint64)
  d_rng_states = random32.create_xoshiro128p_states(num_threads, seed)
  d_global_tally = cuda.to_device(np.zeros(NUM_OUTCOMES, np.int64))
  gpu_bitcount[blocks, THREADS_PER_BLOCK](d_rng_states, decks_per_thread, d_global_tally)
  return d_global_tally.copy_to_host() / (num_threads * decks_per_thread * HANDS_PER_DECK)


# %%
# simulate_poker_hands(10**9, 'bitcount_gpu_cuda', simulate_hands_bitcount_gpu_cuda)  # ~9-11 G hands/s.

# %%
if cuda.is_available():
  assert np.allclose(simulate_hands_bitcount_gpu_cuda(10**7, RNG), EXPECTED_PROB, atol=0.0001)

# %%
if cuda.is_available():
  print('Timing:')
  # %timeit -n1 -r5 simulate_hands_bitcount_gpu_cuda(2 * 10**7, RNG)  # ~85-120 ms.

# %%
if cuda.is_available():
  write_numba_assembly_code(gpu_bitcount, 'gpu_bitcount.ptx')
  report_cuda_kernel_properties(gpu_bitcount)

# %% [markdown]
# ## Approach 3: 52-bit deck bitmask


# %% [markdown]
# Because poker uses a single 52-card deck, we encode each card as a **single bit** within a 52-bit "deck mask" stored in a `uint64`.
#
# We represent a hand of five cards by simpling `OR`-ing together the corresponding five "deck masks".

# %%
# Thanks to Marcel Gavriliu for this approach of storing card deck masks.

# %%
THREADS_PER_BLOCK = 256

# %% [markdown]
# ### GPU simulation


# %%
@numba_njit(inline='always')
def deckmask_of_card(card: numba.uint8) -> numba.uint64:
  """Return a deckmask of 52 bits encoding the suit and rank of `card` (in `range(52)`)."""
  return numba.uint64(1) << numba.uint32(card)


# %%
@cuda.jit(fastmath=True)  # type: ignore[misc]
def gpu_deckmask(rng_states: _CudaArray, decks_per_thread: int, global_tally: _CudaArray) -> None:
  """CUDA kernel to compute hand probabilities using a deck-mask approach."""
  # pylint: disable=too-many-function-args, no-value-for-parameter, comparison-with-callable
  uint8, int32, uint32, float32 = numba.uint8, numba.int32, numba.uint32, numba.float32
  popc = cuda.popc
  thread_index = cuda.grid(1)
  if thread_index >= len(rng_states):
    return
  thread_id = cuda.threadIdx.x  # Index within block.

  # Per-thread hand outcome tallies (ROYAL_FLUSH, ..., HIGH_CARD).
  t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = t8 = t9 = uint32(0)

  block_deck = cuda.shared.array((THREADS_PER_BLOCK, DECK_SIZE), np.uint8)
  deck = block_deck[thread_id]

  for i in range(uint8(DECK_SIZE)):  # Casting as uint8 nicely unrolls the loop.
    deck[i] = i

  rng = rng_states[thread_index]
  s0, s1, s2, s3 = rng['s0'], rng['s1'], rng['s2'], rng['s3']

  for _ in range(int32(decks_per_thread)):
    # Apply Fisher-Yates shuffle to current deck, iteratively swapping [i] with any [0, ..., i].
    i = uint32(51)
    while i > 0:
      random_uint32, s0, s1, s2, s3 = random32.xoshiro128p_next_raw(s0, s1, s2, s3)
      random_uint32 = uint32(random_uint32)
      s0, s1, s2, s3 = uint32(s0), uint32(s1), uint32(s2), uint32(s3)
      j = uint32(float32(random_uint32) * float32((1.0 - 1e-7) / 2**32) * float32(int32(i + 1)))
      deck[i], deck[j] = deck[j], deck[i]
      i = uint32(i - 1)

    mask0, mask1 = deckmask_of_card(deck[0]), deckmask_of_card(deck[1])
    mask2, mask3 = deckmask_of_card(deck[2]), deckmask_of_card(deck[3])
    deckmask = mask0 + mask1 + mask2 + mask3

    card_index = uint32(4)
    while card_index < uint32(DECK_SIZE):
      mask4 = deckmask_of_card(deck[card_index])
      deckmask += mask4  # No bit carries; same as "deckmask |= mask4".

      # Extract four 13-bit masks for the ranks in each suit:
      RANK_MASK = (1 << NUM_RANKS) - 1
      r0 = uint32(uint32(deckmask >> (NUM_RANKS * 0)) & RANK_MASK)
      r1 = uint32(uint32(deckmask >> (NUM_RANKS * 1)) & RANK_MASK)
      r2 = uint32(uint32(deckmask >> (NUM_RANKS * 2)) & RANK_MASK)
      r3 = uint32(uint32(deckmask >> (NUM_RANKS * 3)) & RANK_MASK)
      ranks = uint32(r0 | r1 | r2 | r3)

      is_flush = max(popc(r0), popc(r1), popc(r2), popc(r3)) == CARDS_FOR_A_FLUSH
      bunched_bits = uint32(cuda.clz(ranks) + cuda.ffs(ranks)) == 32 - HAND_SIZE + 1
      is_non_ace_low_straight = (popc(ranks) == CARDS_FOR_A_STRAIGHT) & bunched_bits
      is_ace_low_straight = ranks == 0b1000000001111
      is_straight = is_non_ace_low_straight or is_ace_low_straight
      is_royal_straight = ranks == 0b1111100000000
      fours = uint32(r0 & r1 & r2 & r3)
      three_or_more = uint32((r0 & r1 & r2) | (r0 & r1 & r3) | (r0 & r2 & r3) | (r1 & r2 & r3))
      two_or_more = uint32((r0 & r1) | (r0 & r2) | (r0 & r3) | (r1 & r2) | (r1 & r3) | (r2 & r3))
      num_two_or_more = popc(two_or_more)

      if is_flush and is_straight:
        if is_royal_straight:
          t0 = uint32(t0 + 1)  # ROYAL_FLUSH.
        else:
          t1 = uint32(t1 + 1)  # STRAIGHT_FLUSH.
      elif fours != 0:
        t2 = uint32(t2 + 1)  # FOUR_OF_A_KIND.
      elif three_or_more != 0 and num_two_or_more > 1:
        t3 = uint32(t3 + 1)  # FULL_HOUSE.
      elif is_flush:
        t4 = uint32(t4 + 1)  # FLUSH.
      elif is_straight:
        t5 = uint32(t5 + 1)  # STRAIGHT.
      elif three_or_more != 0:
        t6 = uint32(t6 + 1)  # THREE_OF_A_KIND.
      elif num_two_or_more == 2:
        t7 = uint32(t7 + 1)  # TWO_PAIR.
      elif num_two_or_more == 1:
        t8 = uint32(t8 + 1)  # ONE_PAIR.
      else:
        t9 = uint32(t9 + 1)  # HIGH_CARD.

      deckmask -= mask0  # No bit carries; same as "deckmask &= ~mask0".
      mask0, mask1, mask2, mask3 = mask1, mask2, mask3, mask4
      card_index = uint32(card_index + 1)

  pairs = (0, t0), (1, t1), (2, t2), (3, t3), (4, t4), (5, t5), (6, t6), (7, t7), (8, t8), (9, t9)
  for i, t in pairs:
    cuda.atomic.add(global_tally, i, t)


# %%
def simulate_hands_deckmask_gpu_cuda(num_decks: int, rng: np.random.Generator) -> Probabilities:
  """Compute hand probabilities using a deckmask approach by invoking a CUDA kernel."""
  device = cuda.get_current_device()
  PARALLELISM = 8  # Number of blocks per SM.
  target_num_threads = PARALLELISM * device.MULTIPROCESSOR_COUNT * THREADS_PER_BLOCK
  decks_per_thread = max(1, num_decks // target_num_threads)
  num_threads = num_decks // decks_per_thread
  # print(f'{decks_per_thread=} {num_threads=}')
  blocks = math.ceil(num_threads / THREADS_PER_BLOCK)
  seed = rng.integers(2**64, dtype=np.uint64)
  d_rng_states = random32.create_xoshiro128p_states(num_threads, seed)
  d_global_tally = cuda.to_device(np.zeros(NUM_OUTCOMES, np.int64))
  gpu_deckmask[blocks, THREADS_PER_BLOCK](d_rng_states, decks_per_thread, d_global_tally)
  return d_global_tally.copy_to_host() / (num_threads * decks_per_thread * HANDS_PER_DECK)


# %%
if cuda.is_available():
  assert np.allclose(simulate_hands_deckmask_gpu_cuda(10**7, RNG), EXPECTED_PROB, atol=0.0001)

# %%
if cuda.is_available():
  print('Timing:')
  # %timeit -n1 -r5 simulate_hands_deckmask_gpu_cuda(5 * 10**7, RNG)  # ~65-180 ms.

# %%
if cuda.is_available():
  write_numba_assembly_code(gpu_deckmask, 'gpu_deckmask.ptx')
  report_cuda_kernel_properties(gpu_deckmask)

# %%
# simulate_poker_hands(10**10, 'deckmask_gpu_cuda', simulate_hands_deckmask_gpu_cuda)  # ~17-41 G hands/s.

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
        'bitcount_cpu_numba': simulate_hands_bitcount_cpu_numba,
        'bitcount_cpu_numba_multiprocess': simulate_hands_bitcount_cpu_numba_multiprocess,
        'bitcount_gpu_cuda': simulate_hands_bitcount_gpu_cuda,
        'deckmask_gpu_cuda': simulate_hands_deckmask_gpu_cuda,
    }.items()
    if ('multiprocess' not in name or MULTIPROCESS_IS_AVAILABLE)
    and ('cuda' not in name or cuda.is_available())
}

# %%
COMPLEXITY_ADJUSTMENT = {
    'array_cpu_python': 0.01,
    'array_cpu_numba': 2.0,
    'array_cpu_numba_multiprocess': 20.0,
    'array_gpu_cuda': 500.0,
    'bitcount_cpu_numba': 2.0,
    'bitcount_cpu_numba_multiprocess': 20.0,
    'bitcount_gpu_cuda': 2000.0,
    'deckmask_gpu_cuda': 4000.0,
}


# %%
def simulate_poker_hands(
    desired_num_hands: int,
    func_name: str,
    func: Callable[[int, np.random.Generator], Probabilities],
    verbose: bool = False,
) -> None:
  """Report hand probabilities and timings using a particular approach."""
  num_decks = math.ceil(desired_num_hands / HANDS_PER_DECK)
  num_hands = num_decks * HANDS_PER_DECK

  # Ensure the function is jitted.
  _ = func(num_decks // 10, RNG)

  start_time = time.perf_counter_ns()
  results = func(num_decks, RNG)
  elapsed_time = (time.perf_counter_ns() - start_time) / 10**9

  round_digits: Any = lambda x, ndigits=3: round(x, ndigits - 1 - math.floor(math.log10(abs(x))))
  hands_per_s: float = round_digits(int(num_hands / elapsed_time))
  print(
      f'\n# For {func_name:<32} {num_hands:13,} hands;'
      f' {elapsed_time:.3f} s; {hands_per_s:13,} hands/s.'
  )

  if verbose:
    print('#  Probabilities:')
  sigmas = []
  for outcome, result_prob in zip(Outcome, results):
    reference_prob = outcome.reference_count / comb(DECK_SIZE, HAND_SIZE)
    error = result_prob - reference_prob
    estimate_sdv = (reference_prob * (1 - reference_prob) / num_hands) ** 0.5
    sdv = error / estimate_sdv
    sigmas.append(f' {sdv:6.2f}σ')
    s = f'#   {outcome.string_name:<16}: {result_prob * 100:8.5f}%'
    s += f'  (vs. ref. {reference_prob * 100:8.5f}%  error:{error * 100:8.5f}% {sdv:6.2f}σ)'
    if verbose:
      print(s)
  if not verbose:
    print('#   ' + ''.join(sigmas))


# %%
def compare_simulations(base_num_hands: int, verbose: bool = False) -> None:
  """Report hand probabilities and timings using a variety of approaches on CPU and GPU."""
  for func_name, func in SIMULATE_FUNCTIONS.items():
    desired_num_hands = math.ceil(base_num_hands * COMPLEXITY_ADJUSTMENT[func_name])
    simulate_poker_hands(desired_num_hands, func_name, func, verbose)


# %%
compare_simulations(base_num_hands=10**7)

# %%
# Hands/s: 135k, 44m, 470m, 16g, 135m, 890m, 57g, 108g.

# %%
if 1:
  if cuda.is_available():
    simulate_poker_hands(10**12, 'bitcount_gpu_cuda', simulate_hands_bitcount_gpu_cuda)

# %%
# For bitcount_gpu_cuda                1,000,000,000,032 hands; 14.303 s; 69,900,000,000 hands/s.
#      0.37σ  -0.78σ   1.09σ  -0.53σ  -0.53σ  -0.72σ  -1.61σ  -0.95σ  -0.91σ   1.91σ

# %%
if 1:
  if cuda.is_available():
    simulate_poker_hands(10**12, 'deckmask_gpu_cuda', simulate_hands_deckmask_gpu_cuda)

# %%
# For deckmask_gpu_cuda                1,000,000,000,032 hands; 6.777 s; 148,000,000,000 hands/s.
#     -0.07σ  -1.39σ   1.84σ   1.33σ   0.81σ  -1.17σ  -1.01σ  -0.69σ  -0.40σ   0.91σ

# %% [markdown]
# ## End
