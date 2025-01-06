# %%
USE_UINT_RANDOM = True  # Using uint remainder is faster than float truncation.

if USE_UINT_RANDOM:  # Faster and has lower bias (~2.76e-18 for worst case ).
  random_uint32 = random_next_uniform_uint(rng_states, thread_index)
  j = random_uint32 % numba.uint32(i + 1)
else:  # Results in higher bias (~2.86e-6) due to reduced size (24 bits) of mantissa.
  j = int(random_next_uniform_float32(rng_states, thread_index) * (i + 1))

# %%
assert ROYAL_STRAIGHT_RANKS == int(sum(2 ** np.arange(NUM_RANKS - CARDS_FOR_A_STRAIGHT, NUM_RANKS)))


# %%
def create_table_flush_from_suit_count_mask():
  table = np.full(2 ** (NUM_SUITS * CARD_COUNT_BITS), False)
  for nums in itertools.product(range(NUM_CARD_COUNT), repeat=NUM_SUITS):
    if max(nums) >= CARDS_FOR_A_FLUSH:
      mask = np.dot(nums, NUM_CARD_COUNT ** np.arange(NUM_SUITS))
      table[mask] = True
  return table


# %%
TABLE_FLUSH_FROM_SUIT_COUNT_MASK = create_table_flush_from_suit_count_mask()

# %%
TABLE_FLUSH = cuda.const.array_like(TABLE_FLUSH_FROM_SUIT_COUNT_MASK)
is_flush = TABLE_FLUSH[suit_count_mask]


# %%
@numba.njit
def get_ranks_present(rank_count_mask):
  """Return a (contiguous) 13-bit mask that encodes which ranks contain 1-3 cards each."""
  x = (rank_count_mask | (rank_count_mask >> 1)) & RANKS_ONE
  # Combine groups by progressively larger shifts.  https://stackoverflow.com/a/28358035
  x = (x | (x >> 2)) & 0x30C30C30C30C30C3
  x = (x | (x >> 4)) & 0xF00F00F00F00F00F
  x = (x | (x >> 8)) & 0x00FF0000FF0000FF
  x = (x | (x >> 16)) & 0xFFFF00000000FFFF
  x = (x | (x >> 32)) & 0x00000000FFFFFFFF
  # assert not (x & ~0x1fff)
  return x


# %%
def create_table_straight_from_rank_present():
  table = np.full(2**NUM_RANKS, False)
  for i in range(9):  # Straights from (0, 1, 2, 3, 4) '23456' to (8, 9, 10, 11, 12) 'TJQKA'.
    table[0b11111 << i] = True
  table[0b1000000001111] = True  # (0, 1, 2, 3, 12) for ace-low straight.
  return table


# %%
TABLE_STRAIGHT_FROM_RANK_PRESENT = create_table_straight_from_rank_present()

# %%
TABLE_STRAIGHT = cuda.const.array_like(TABLE_STRAIGHT_FROM_RANK_PRESENT)
is_straight = TABLE_STRAIGHT[rank_present]

# %%
TABLE_STRAIGHTS = cuda.const.array_like(TABLE_STRAIGHTS_RANK_PRESENT)


# %%
def create_table_straights_rank_present():
  lst = [0b11111 << i for i in range(9)]  # (0, 1, 2, 3, 4) '23456' to (8, 9, 10, 11, 12) 'TJQKA'.
  lst.append(0b1000000001111)  # (0, 1, 2, 3, 12) for ace-low straight.
  return np.array(lst, np.uint16)


# %%
# Table of 10 uint32, each the 13-bit mask encoding of a straight.
TABLE_STRAIGHTS_RANK_PRESENT = create_table_straights_rank_present()

# %%
is_straight = rank_present in TABLE_STRAIGHTS_RANK_PRESENT
for straight in TABLE_STRAIGHTS_RANK_PRESENT:  # Even faster than cuda.const.array_like().
  if rank_present == straight:
    is_straight = True

# %%
mask_pairs_bits = rank_count_mask & (RANKS_TWO | RANKS_FOUR)
mask_pairs = (mask_pairs_bits | (mask_pairs_bits >> 1)) & RANKS_TWO
num_two_or_more = numba.uint32(cuda.popc(mask_pairs))  # Count number of set bits.

# %%
ROYAL_STRAIGHT_RANKS = 0b1111100000000
rank_present = numba.uint16(get_ranks_present(rank_count_mask))  # Valid only if not is_four.

if (rank_present & ROYAL_STRAIGHT_RANKS) != 0:
  return Outcome.ROYAL_FLUSH.value

# %%
import numba.cpython.mathimpl
import numba.extending


@numba.extending.intrinsic
def popc_helper(typing_context, src):
  _ = typing_context
  sig = numba.uint64(numba.uint64)

  def codegen(context, builder, signature, args):
    return numba.cpython.mathimpl.call_fp_intrinsic(builder, "llvm.ctpop.i64", args)

  return sig, codegen


@numba.njit(numba.uint64(numba.uint64))
def popc(x):
  """Return the (population) count of set bits in an integer."""
  # https://stackoverflow.com/a/77103233
  return popc_helper(x)


print(popc(43))


# %%
# for mask in [0b_101_000_000_000, 0b_100_001_000_0000, 0b_011_010_000_000, 0b_011_001_001_000, 0b_010_010_001_000, 0b_010_001_001_001]:
#   expr1 = popc(mask & (mask >> 2))
#   t = mask & (SUITS_ONE | (SUITS_ONE << 2))
#   expr2 = popc(t & (t >> 2))
#   expr3 = popc(t & (t >> 2) & (SUITS_ONE))
#   print(mask, popc(mask), popc(mask + SUITS_ONE), popc(mask + SUITS_ONE + SUITS_ONE), expr1, expr2, expr3)

# 2560 2 5 6 1 1 1
# 4224 2 6 5 0 0 0
# 1664 3 5 5 1 0 0
# 1608 4 4 7 0 0 0
# 1160 3 6 5 0 0 0
# 1097 4 5 7 0 0 0

# %%
def write_cuda_assembly_code():
  if 0:
    signature = (cuda.random.xoroshiro128p_type[:], numba.int64, numba.int64[:])
    ptx, _ = cuda.compile_ptx_for_current_device(compute_gpu, signature, device=True, abi='c')
  else:
    (ptx,) = compute_gpu.inspect_asm().values()
  pathlib.Path('compute_gpu.ptx').write_text(ptx, encoding='utf-8')


# %%
@cuda.jit(numba.uint16(numba.uint64), device=True, fastmath=True)
