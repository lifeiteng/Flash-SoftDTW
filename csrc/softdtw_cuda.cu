// Copyright (c) 2026 Feiteng Li. Licensed under the Apache License, Version 2.0.
//
// CUDA backend of Flash-SoftDTW.
//
// Parallelization scheme ("skewed strip wavefront"):
//
//   * Rows of one problem are processed in horizontal strips of P rows, one
//     thread per row. Inside a strip, thread k (row i = i0 + k) handles column
//     j = jbase + t - k at tick t, so all three DP predecessors of a cell were
//     produced at ticks t-1 / t-2 by the same thread or by thread k-1.
//   * Neighbour values travel through warp shuffles; only the boundary between
//     two warps goes through shared memory (one barrier per tick), and only the
//     boundary between two strips goes through a small per-problem column
//     buffer in global memory. Nothing of size T x S is kept in shared memory.
//   * With P == 32 (one warp per strip) a tick needs no barrier at all, and a
//     thread block packs several independent problems, one per warp. This is
//     the fast path for narrow bands and large batches.
//   * Softmin is evaluated in base 2 (exp2 / log2), which maps to the hardware
//     MUFU instructions for fp32.
//   * The forward pass keeps R only when gradients are required. The backward
//     pass reuses the same wavefront in reverse tick order and propagates the
//     expected alignment as messages, so it needs no transition-probability
//     tensors: it recomputes them from R and cost on the fly.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>

#include "softdtw.h"

namespace flash_softdtw {
namespace {

constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

template <typename T>
struct AccMath;

template <>
struct AccMath<float> {
  // Hardware MUFU approximations (2 ulp), independent of --use_fast_math.
  static __device__ __forceinline__ float exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }
  static __device__ __forceinline__ float log2(float x) {
    float y;
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  }
  static __device__ __forceinline__ float inf() { return __int_as_float(0x7f800000); }
  static __device__ __forceinline__ float div(float a, float b) { return __fdividef(a, b); }
};

template <>
struct AccMath<double> {
  static __device__ __forceinline__ double exp2(double x) { return ::exp2(x); }
  static __device__ __forceinline__ double log2(double x) { return ::log2(x); }
  static __device__ __forceinline__ double inf() {
    return __longlong_as_double(0x7ff0000000000000ULL);
  }
  static __device__ __forceinline__ double div(double a, double b) { return a / b; }
};

template <typename acc_t>
struct Params {
  // Cost tensor [B, T, W] with arbitrary element strides.
  const void* cost;
  int64_t cost_sb, cost_st, cost_sw;
  int W;
  // Per-problem lengths (int32, contiguous).
  const int* x_lens;
  const int* y_lens;
  // Optional band [B, Tmax] (int32, arbitrary strides, e.g. two views of a
  // packed [B, T, 2] tensor). nullptr means dense mode.
  const int* begin;
  const int* end;
  int64_t begin_sb, begin_st, end_sb, end_st;
  int Tmax;
  int B;
  // Accumulators.
  acc_t* r;             // [B, Tmax, W] or nullptr
  acc_t* values;        // [B]
  acc_t* edge_a;        // [B, W] strip-boundary scratch, indexed by j - off
  acc_t* edge_b;        // [B, W] scratch (backward only)
  const acc_t* grad_values;  // [B] (backward only)
  void* grad_cost;      // [B, Tmax, W] contiguous, same dtype as cost
  acc_t gamma;
  acc_t warp;
  acc_t scale;          // log2(e) / gamma
};

// Band of one row in global column coordinates, clipped to [0, S) and to the
// columns that exist in the cost tensor. Returns the storage offset in `off`.
template <typename acc_t>
__device__ __forceinline__ void row_band(const Params<acc_t>& p, int b, int i,
                                         int T, int S, int& lo, int& hi,
                                         int& off) {
  if (i < 0 || i >= T) {
    lo = 0;
    hi = 0;
    off = 0;
    return;
  }
  if (p.begin == nullptr) {
    off = 0;
    lo = 0;
    hi = min(S, p.W);
    return;
  }
  off = p.begin[b * p.begin_sb + i * p.begin_st];
  lo = max(off, 0);
  hi = min(min(p.end[b * p.end_sb + i * p.end_st], S), off + p.W);
}

// Unreachable cells hold a large finite sentinel instead of +inf: softmin then
// needs no special cases (exp2 of a hugely negative number flushes to zero and
// BIG + small == BIG), which keeps the per-tick dependency chain branch-free.
template <typename acc_t>
struct Sentinel;
template <>
struct Sentinel<float> {
  static constexpr float big = 1e30f;
  static constexpr float reachable_below = 1e29f;
};
template <>
struct Sentinel<double> {
  static constexpr double big = 1e300;
  static constexpr double reachable_below = 1e299;
};

// All recursions run in base-2 units: R' = R * log2(e) / gamma. Then
//   R'(i, j) = C'(i, j) + m - log2(2^(m-a) + 2^(m-b) + 2^(m-c)),  m = min(a,b,c)
// which needs two MUFU stages and no multiply or divide on the critical path.
template <typename acc_t>
__device__ __forceinline__ acc_t softmin3_log2(acc_t a, acc_t b, acc_t c) {
  using M = AccMath<acc_t>;
  const acc_t m = fmin(a, fmin(b, c));
  return m - M::log2(M::exp2(m - a) + M::exp2(m - b) + M::exp2(m - c));
}

template <typename T>
__device__ __forceinline__ T warp_max(T v) {
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(kFullMask, v, o));
  return v;
}

// Number of ticks a strip needs: max over rows of (k + hi_k - 1 - jbase) + 1.
template <bool kMultiWarp>
__device__ __forceinline__ int strip_ticks(int local_last, int* red) {
  int v = warp_max(local_last);
  if (!kMultiWarp) return v + 1;
  const int lane = threadIdx.x & (kWarp - 1);
  const int wid = threadIdx.x / kWarp;
  if (lane == 0) red[wid] = v;
  __syncthreads();
  if (wid == 0) {
    const int nw = blockDim.x / kWarp;
    int x = lane < nw ? red[lane] : -1;
    x = warp_max(x);
    if (lane == 0) red[kWarp] = x;
  }
  __syncthreads();
  const int r = red[kWarp] + 1;
  __syncthreads();
  return r;
}

constexpr int kWarpsPerBlock = 4;  // problems per block in single-warp mode
constexpr int kPrefetch = 8;       // ticks of operands loaded ahead

template <typename scalar_t, typename acc_t, bool kMultiWarp, bool kStoreR>
__global__ void __launch_bounds__(kMultiWarp ? 1024 : kWarpsPerBlock * 32)
softdtw_forward_kernel(const Params<acc_t> p) {
  using M = AccMath<acc_t>;
  const acc_t INF = M::inf();
  const acc_t BIG = Sentinel<acc_t>::big;
  const int lane = threadIdx.x & (kWarp - 1);
  const int wid = threadIdx.x / kWarp;

  int b, k, P;
  if (kMultiWarp) {
    b = blockIdx.x;
    k = threadIdx.x;
    P = blockDim.x;
  } else {
    b = blockIdx.x * (blockDim.x / kWarp) + wid;
    k = lane;
    P = kWarp;
  }
  if (b >= p.B) return;  // uniform per warp (single-warp) or per block

  __shared__ acc_t xfer[2][kWarp];
  __shared__ int red[kWarp + 1];

  const int T = p.x_lens[b];
  const int S = p.y_lens[b];
  if (T <= 0 || S <= 0) {
    if (k == 0) p.values[b] = (T <= 0 && S <= 0) ? acc_t(0) : INF;
    return;
  }
  if (k == 0) p.values[b] = INF;
  if (T > p.Tmax || (p.begin == nullptr && S > p.W)) return;

  const scalar_t* cost = static_cast<const scalar_t*>(p.cost) + b * p.cost_sb;
  acc_t* edge = p.edge_a + static_cast<int64_t>(b) * p.W;
  acc_t* rmat = kStoreR ? p.r + static_cast<int64_t>(b) * p.Tmax * p.W : nullptr;
  const acc_t sc = p.scale;           // log2(e) / gamma
  const acc_t wp = p.warp * sc;       // warp penalty in base-2 units
  const acc_t to_nat = p.gamma * acc_t(0.69314718055994530942);
  const int last_lane = kMultiWarp ? (blockDim.x - 1) : (kWarp - 1);

  for (int i0 = 0; i0 < T; i0 += P) {
    const int i = i0 + k;
    int lo, hi, off;
    row_band(p, b, i, T, S, lo, hi, off);
    int jlo0, jhi0, joff0;
    row_band(p, b, i0, T, S, jlo0, jhi0, joff0);
    const int jbase = jlo0;
    int plo = 0, phi = 0, poff = 0;
    if (k == 0 && i0 > 0) row_band(p, b, i0 - 1, T, S, plo, phi, poff);

    const int local_last = (hi > lo) ? (k + hi - 1 - jbase) : -1;
    const int ticks = strip_ticks<kMultiWarp>(local_last, red);

    if (kMultiWarp) {
      if (threadIdx.x < 2 * kWarp) (&xfer[0][0])[threadIdx.x] = BIG;
      __syncthreads();
    }

    // `prev_up` is the value received as "up" at the previous tick; for every
    // thread it equals the diagonal predecessor R'(i-1, j-1) of the current
    // tick. The first row of the strip seeds it from the row above.
    acc_t prev_up = BIG;
    if (k == 0) {
      const int c = jbase - 1;
      if (i0 == 0) {
        prev_up = (c == -1) ? acc_t(0) : BIG;
      } else if (c >= plo && c < phi) {
        prev_up = edge[c - poff];
      }
    }
    acc_t v1 = BIG;  // own value at tick t-1 = R'(i, j-1)

    const scalar_t* crow = cost + static_cast<int64_t>(i) * p.cost_st;
    const bool fetch_edge = (k == 0) && (i0 > 0);

    // Software pipeline: operands of the next kPrefetch ticks are loaded while
    // the current ones are consumed.
    acc_t cbuf[kPrefetch], ebuf[kPrefetch];
    auto fetch = [&](int t0, acc_t* cb, acc_t* eb) {
#pragma unroll
      for (int u = 0; u < kPrefetch; ++u) {
        const int j = jbase + t0 + u - k;
        // Clamping keeps +-inf costs (obstacles) from turning into inf - inf.
        cb[u] = (j >= lo && j < hi)
                    ? fmax(fmin(static_cast<acc_t>(crow[static_cast<int64_t>(j - off) * p.cost_sw]) * sc,
                                BIG), -BIG)
                    : acc_t(0);
        eb[u] = (fetch_edge && j >= plo && j < phi) ? edge[j - poff] : BIG;
      }
    };
    fetch(0, cbuf, ebuf);

    for (int t0 = 0; t0 < ticks; t0 += kPrefetch) {
      acc_t cnext[kPrefetch], enext[kPrefetch];
      fetch(t0 + kPrefetch, cnext, enext);
#pragma unroll
      for (int u = 0; u < kPrefetch; ++u) {
        const int t = t0 + u;
        if (t >= ticks) break;  // uniform across the strip
        const int j = jbase + t - k;
        acc_t up = __shfl_up_sync(kFullMask, v1, 1);
        if (kMultiWarp) {
          const acc_t xv = xfer[(t - 1) & 1][wid > 0 ? wid - 1 : 0];
          up = (lane == 0) ? xv : up;
        }
        up = (k == 0) ? ebuf[u] : up;
        const acc_t diag = prev_up;
        prev_up = up;

        const bool active = (j >= lo) && (j < hi);
        // A cell is reachable only through a reachable predecessor: without
        // this, a huge negative cost could pull BIG back below the threshold.
        const acc_t pmin = fmin(diag, fmin(up + wp, v1 + wp));
        const bool reach = active && (pmin < Sentinel<acc_t>::reachable_below);
        const acc_t val =
            reach ? fmin(cbuf[u] + softmin3_log2(diag, up + wp, v1 + wp), BIG) : BIG;
        if (active) {
          if (kStoreR) rmat[static_cast<int64_t>(i) * p.W + (j - off)] = val;
          if (i == T - 1 && j == S - 1) p.values[b] = reach && val < BIG ? val * to_nat : INF;
          if (k == last_lane && i + 1 < T) edge[j - off] = val;
        }
        if (kMultiWarp) {
          if (lane == kWarp - 1) xfer[t & 1][wid] = val;
          __syncthreads();
        }
        v1 = val;
      }
#pragma unroll
      for (int u = 0; u < kPrefetch; ++u) {
        cbuf[u] = cnext[u];
        ebuf[u] = enext[u];
      }
    }
    if (kMultiWarp) {
      __syncthreads();
    } else {
      __syncwarp();
    }
  }
}

template <typename scalar_t, typename acc_t, bool kMultiWarp>
__global__ void __launch_bounds__(kMultiWarp ? 1024 : kWarpsPerBlock * 32)
softdtw_backward_kernel(const Params<acc_t> p) {
  using M = AccMath<acc_t>;
  const acc_t BIG = Sentinel<acc_t>::big;
  const int lane = threadIdx.x & (kWarp - 1);
  const int wid = threadIdx.x / kWarp;

  int b, k, P;
  if (kMultiWarp) {
    b = blockIdx.x;
    k = threadIdx.x;
    P = blockDim.x;
  } else {
    b = blockIdx.x * (blockDim.x / kWarp) + wid;
    k = lane;
    P = kWarp;
  }
  if (b >= p.B) return;

  // Messages from lane 0 of warp w+1 to lane 31 of warp w: {up, diag}.
  __shared__ acc_t xup[2][kWarp];
  __shared__ acc_t xdg[2][kWarp];
  __shared__ int red[kWarp + 1];

  const int T = p.x_lens[b];
  const int S = p.y_lens[b];
  if (T <= 0 || S <= 0 || T > p.Tmax || (p.begin == nullptr && S > p.W)) return;

  scalar_t* gcost = static_cast<scalar_t*>(p.grad_cost) +
                    static_cast<int64_t>(b) * p.Tmax * p.W;
  const acc_t* rmat = p.r + static_cast<int64_t>(b) * p.Tmax * p.W;
  // Messages published by the first row of a strip, keyed by the column of
  // the sending cell relative to that row's storage offset.
  acc_t* e_up = p.edge_a + static_cast<int64_t>(b) * p.W;
  acc_t* e_dg = p.edge_b + static_cast<int64_t>(b) * p.W;
  const acc_t seed = p.grad_values[b];
  const acc_t wp = p.warp * p.scale;
  const acc_t HALF = Sentinel<acc_t>::reachable_below;
  const int last_lane = kMultiWarp ? (blockDim.x - 1) : (kWarp - 1);
  const int nstrips = (T + P - 1) / P;

  for (int s = nstrips - 1; s >= 0; --s) {
    const int i0 = s * P;
    const int i = i0 + k;
    int lo, hi, off;
    row_band(p, b, i, T, S, lo, hi, off);
    int jlo0, jhi0, joff0;
    row_band(p, b, i0, T, S, jlo0, jhi0, joff0);
    const int jbase = jlo0;
    int plo = 0, phi = 0, poff = 0;
    if (lane == 0) row_band(p, b, i - 1, T, S, plo, phi, poff);
    int nlo = 0, nhi = 0, noff = 0;
    if (k == last_lane) row_band(p, b, i + 1, T, S, nlo, nhi, noff);

    const int local_last = (hi > lo) ? (k + hi - 1 - jbase) : -1;
    const int ticks = strip_ticks<kMultiWarp>(local_last, red);

    if (kMultiWarp) {
      if (threadIdx.x < 2 * kWarp) {
        (&xup[0][0])[threadIdx.x] = acc_t(0);
        (&xdg[0][0])[threadIdx.x] = acc_t(0);
      }
      __syncthreads();
    }

    const acc_t* rrow = rmat + static_cast<int64_t>(i) * p.W;
    const acc_t* rprev = rmat + static_cast<int64_t>(i - 1) * p.W;
    auto load_r = [&](int jj) -> acc_t {
      return (jj >= lo && jj < hi) ? rrow[jj - off] : BIG;
    };

    // Sliding window of own values: cur = R'(i, j), left = R'(i, j-1),
    // left2 = R'(i, j-2) for the column j of the current tick.
    const int t_start = ticks - 1;
    const int j_start = jbase + t_start - k;
    acc_t cur = load_r(j_start);
    acc_t left = load_r(j_start - 1);
    acc_t left2 = load_r(j_start - 2);

    // `prev_dg` is the diagonal predecessor seen at the previous (higher) tick;
    // it equals the "up" predecessor R'(i-1, j) of the current tick.
    acc_t prev_dg = __shfl_up_sync(kFullMask, left, 1);
    const bool is_first = (lane == 0) && (i > 0);
    const bool is_last = (k == last_lane) && (i + 1 < T);
    if (lane == 0) {
      prev_dg = (is_first && j_start >= plo && j_start < phi) ? rprev[j_start - poff] : BIG;
    }

    acc_t left_in = 0;   // message from (i, j+1)
    acc_t up_sent = 0;   // own up message of the previous tick
    acc_t dg_sent = 0;   // own diag message of the previous tick
    acc_t dg_recv = 0;   // neighbour's diag message, consumed one tick later
    acc_t dg_hold = 0;   // multi-warp lane 0: diag message delayed by one tick

    // Software pipeline: three independent load streams.
    //   ra: own row R'(i, j-3)            (refills the sliding window)
    //   rb: lane 0 -> R'(i-1, j-1);  last thread -> e_up message for column j
    //   rc: last thread -> e_dg message for column j
    acc_t ra[kPrefetch], rb[kPrefetch], rc[kPrefetch];
    auto fetch = [&](int tc, acc_t* a, acc_t* bb, acc_t* c) {
#pragma unroll
      for (int u = 0; u < kPrefetch; ++u) {
        const int j = jbase + (tc - u) - k;
        a[u] = load_r(j - 3);
        if (lane == 0) {
          bb[u] = is_first ? ((j - 1 >= plo && j - 1 < phi) ? rprev[j - 1 - poff] : BIG)
                           : ((i == 0 && j == 0) ? acc_t(0) : BIG);
          c[u] = 0;
        } else if (is_last) {
          bb[u] = (j >= nlo && j < nhi) ? e_up[j - noff] : acc_t(0);
          c[u] = (j + 1 >= nlo && j + 1 < nhi) ? e_dg[j + 1 - noff] : acc_t(0);
        } else {
          bb[u] = 0;
          c[u] = 0;
        }
      }
    };
    fetch(t_start, ra, rb, rc);

    for (int tc = t_start; tc >= 0; tc -= kPrefetch) {
      acc_t na[kPrefetch], nb[kPrefetch], nc[kPrefetch];
      fetch(tc - kPrefetch, na, nb, nc);
#pragma unroll
      for (int u = 0; u < kPrefetch; ++u) {
        const int t = tc - u;
        if (t < 0) break;  // uniform across the strip
        const int j = jbase + t - k;

        // Predecessors: diag R'(i-1, j-1) from thread k-1 (or memory), up is
        // last tick's diag.
        acc_t dg_r = __shfl_up_sync(kFullMask, left2, 1);
        dg_r = (lane == 0) ? rb[u] : dg_r;
        const acc_t up_r = prev_dg;
        prev_dg = dg_r;

        // Incoming messages from row i+1.
        const acc_t up_nb = __shfl_down_sync(kFullMask, up_sent, 1);
        const acc_t dg_nb = __shfl_down_sync(kFullMask, dg_sent, 1);
        acc_t up_in = up_nb;
        acc_t dg_in = dg_recv;
        dg_recv = dg_nb;
        if (kMultiWarp && lane == kWarp - 1 && k != last_lane) {
          up_in = xup[(t + 1) & 1][wid + 1];
          dg_in = xdg[(t + 1) & 1][wid + 1];
        }
        if (k == last_lane) {
          up_in = is_last ? rb[u] : acc_t(0);
          dg_in = is_last ? rc[u] : acc_t(0);
        }

        const bool active = (j >= lo) && (j < hi);
        const bool reach = active && (cur < HALF);
        acc_t e = left_in + up_in + dg_in;
        if (i == T - 1 && j == S - 1) e += seed;
        e = reach ? e : acc_t(0);

        // Transition probabilities from the predecessors only.
        const acc_t a = dg_r;
        const acc_t uu = up_r + wp;
        const acc_t l = left + wp;
        const acc_t m = fmin(a, fmin(uu, l));
        const acc_t wa = M::exp2(m - a);
        const acc_t wu = M::exp2(m - uu);
        const acc_t wl = M::exp2(m - l);
        const acc_t norm = M::div(e, wa + wu + wl);
        const acc_t m_dg = wa * norm;
        const acc_t m_up = wu * norm;
        const acc_t m_left = wl * norm;

        if (active) {
          gcost[static_cast<int64_t>(i) * p.W + (j - off)] = static_cast<scalar_t>(e);
          if (k == 0 && i > 0) {
            // First row of the strip publishes messages for the strip above.
            e_up[j - off] = m_up;
            e_dg[j - off] = m_dg;
          }
        }

        if (kMultiWarp) {
          if (lane == 0 && wid > 0) {
            xup[t & 1][wid] = m_up;
            xdg[t & 1][wid] = dg_hold;
            dg_hold = m_dg;
          }
          __syncthreads();
        }

        left_in = m_left;
        up_sent = m_up;
        dg_sent = m_dg;
        cur = left;
        left = left2;
        left2 = ra[u];
      }
#pragma unroll
      for (int u = 0; u < kPrefetch; ++u) {
        ra[u] = na[u];
        rb[u] = nb[u];
        rc[u] = nc[u];
      }
    }
    if (kMultiWarp) {
      __syncthreads();
    } else {
      __syncwarp();
    }
  }
}

struct Launch {
  bool multi_warp;
  int threads;
  int blocks;
};

Launch choose_launch(int B, int Tmax, int W, bool banded, int64_t rows_per_block) {
  int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int P;
  if (rows_per_block > 0) {
    P = static_cast<int>(rows_per_block);
  } else {
    // Latency-optimal strip height covers the whole sequence (up to 1024);
    // shrink it while the grid would oversubscribe the device, and keep it
    // close to the band width for banded problems.
    int nw = std::min(32, (Tmax + kWarp - 1) / kWarp);
    const int64_t capacity = static_cast<int64_t>(sms) * 1024;
    while (nw > 1 && static_cast<int64_t>(B) * nw * kWarp > capacity) nw = (nw + 1) / 2;
    // Narrow bands: one warp per problem (no block barrier per tick) wins.
    if (banded) nw = (W <= 2 * kWarp) ? 1 : std::min(nw, (W + kWarp - 1) / kWarp);
    P = std::max(1, nw) * kWarp;
  }
  P = std::max(kWarp, std::min(1024, (P / kWarp) * kWarp));
  Launch l;
  if (P == kWarp) {
    const int per_block = kWarpsPerBlock;
    l.multi_warp = false;
    l.threads = per_block * kWarp;
    l.blocks = (B + per_block - 1) / per_block;
  } else {
    l.multi_warp = true;
    l.threads = P;
    l.blocks = B;
  }
  return l;
}

struct Prepared {
  at::Tensor x_lens, y_lens, begin, end;
  int B, Tmax, W;
};

Prepared prepare(const at::Tensor& cost, const at::Tensor& x_lens,
                 const at::Tensor& y_lens,
                 const c10::optional<at::Tensor>& begin,
                 const c10::optional<at::Tensor>& end) {
  TORCH_CHECK(cost.is_cuda(), "cost must be a CUDA tensor");
  TORCH_CHECK(cost.dim() == 3, "cost must have shape [B, T, W]");
  Prepared pr;
  pr.B = static_cast<int>(cost.size(0));
  pr.Tmax = static_cast<int>(cost.size(1));
  pr.W = static_cast<int>(cost.size(2));
  TORCH_CHECK(x_lens.dim() == 1 && x_lens.size(0) == pr.B, "x_lens must have shape [B]");
  TORCH_CHECK(y_lens.dim() == 1 && y_lens.size(0) == pr.B, "y_lens must have shape [B]");
  auto iopt = cost.options().dtype(at::kInt);
  pr.x_lens = x_lens.to(iopt).contiguous();
  pr.y_lens = y_lens.to(iopt).contiguous();
  TORCH_CHECK(begin.has_value() == end.has_value(), "begin and end must be given together");
  if (begin.has_value()) {
    TORCH_CHECK(begin->dim() == 2 && begin->size(0) == pr.B && begin->size(1) == pr.Tmax,
                "band begin must have shape [B, T]");
    TORCH_CHECK(end->sizes() == begin->sizes(), "band end must have shape [B, T]");
    // Strided int32 views (e.g. band[..., 0]) are used in place, without a copy.
    pr.begin = begin->to(iopt);
    pr.end = end->to(iopt);
  }
  return pr;
}

template <typename acc_t>
Params<acc_t> base_params(const at::Tensor& cost, const Prepared& pr,
                          double gamma, double warp) {
  Params<acc_t> p{};
  p.cost = cost.data_ptr();
  p.cost_sb = cost.stride(0);
  p.cost_st = cost.stride(1);
  p.cost_sw = cost.stride(2);
  p.W = pr.W;
  p.x_lens = pr.x_lens.data_ptr<int>();
  p.y_lens = pr.y_lens.data_ptr<int>();
  p.begin = pr.begin.defined() ? pr.begin.data_ptr<int>() : nullptr;
  p.end = pr.end.defined() ? pr.end.data_ptr<int>() : nullptr;
  if (pr.begin.defined()) {
    p.begin_sb = pr.begin.stride(0);
    p.begin_st = pr.begin.stride(1);
    p.end_sb = pr.end.stride(0);
    p.end_st = pr.end.stride(1);
  }
  p.Tmax = pr.Tmax;
  p.B = pr.B;
  p.gamma = static_cast<acc_t>(gamma);
  p.warp = static_cast<acc_t>(warp);
  p.scale = static_cast<acc_t>(1.4426950408889634 / gamma);
  return p;
}

#define FLASH_SOFTDTW_DISPATCH(TYPE, NAME, ...)                                 \
  [&] {                                                                         \
    switch (TYPE) {                                                             \
      case at::kFloat: {                                                        \
        using scalar_t = float;                                                 \
        using acc_t = float;                                                    \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::kDouble: {                                                       \
        using scalar_t = double;                                                \
        using acc_t = double;                                                   \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::kHalf: {                                                         \
        using scalar_t = at::Half;                                              \
        using acc_t = float;                                                    \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::kBFloat16: {                                                     \
        using scalar_t = at::BFloat16;                                          \
        using acc_t = float;                                                    \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      default:                                                                  \
        TORCH_CHECK(false, NAME, ": unsupported dtype ", TYPE);                 \
    }                                                                           \
  }()

}  // namespace

std::vector<at::Tensor> forward_cuda(const at::Tensor& cost,
                                     const at::Tensor& x_lens,
                                     const at::Tensor& y_lens,
                                     const c10::optional<at::Tensor>& begin,
                                     const c10::optional<at::Tensor>& end,
                                     double gamma, double warp, bool save_r,
                                     int64_t rows_per_block) {
  TORCH_CHECK(std::isfinite(gamma) && gamma > 0, "gamma must be finite and positive");
  TORCH_CHECK(std::isfinite(warp), "warp must be finite");
  const c10::cuda::CUDAGuard guard(cost.device());
  Prepared pr = prepare(cost, x_lens, y_lens, begin, end);
  const auto acc_type =
      cost.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  auto aopt = cost.options().dtype(acc_type);
  at::Tensor values = at::empty({pr.B}, aopt);  // initialized by the kernel
  at::Tensor r = save_r ? at::empty({pr.B, pr.Tmax, pr.W}, aopt) : at::Tensor();
  at::Tensor edge = at::empty({pr.B, std::max(1, pr.W)}, aopt);
  if (pr.B == 0) return {values, r};
  const Launch l = choose_launch(pr.B, pr.Tmax, pr.W, begin.has_value(), rows_per_block);
  auto stream = at::cuda::getCurrentCUDAStream();

  FLASH_SOFTDTW_DISPATCH(cost.scalar_type(), "softdtw_forward", [&] {
    Params<acc_t> p = base_params<acc_t>(cost, pr, gamma, warp);
    p.values = values.data_ptr<acc_t>();
    p.edge_a = edge.data_ptr<acc_t>();
    p.r = save_r ? r.data_ptr<acc_t>() : nullptr;
    if (l.multi_warp) {
      if (save_r)
        softdtw_forward_kernel<scalar_t, acc_t, true, true><<<l.blocks, l.threads, 0, stream>>>(p);
      else
        softdtw_forward_kernel<scalar_t, acc_t, true, false><<<l.blocks, l.threads, 0, stream>>>(p);
    } else {
      if (save_r)
        softdtw_forward_kernel<scalar_t, acc_t, false, true><<<l.blocks, l.threads, 0, stream>>>(p);
      else
        softdtw_forward_kernel<scalar_t, acc_t, false, false><<<l.blocks, l.threads, 0, stream>>>(p);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {values, r};
}

at::Tensor backward_cuda(const at::Tensor& cost, const at::Tensor& r,
                         const at::Tensor& x_lens, const at::Tensor& y_lens,
                         const c10::optional<at::Tensor>& begin,
                         const c10::optional<at::Tensor>& end,
                         const at::Tensor& grad_values, double gamma,
                         double warp, int64_t rows_per_block) {
  TORCH_CHECK(std::isfinite(gamma) && gamma > 0, "gamma must be finite and positive");
  TORCH_CHECK(std::isfinite(warp), "warp must be finite");
  const c10::cuda::CUDAGuard guard(cost.device());
  Prepared pr = prepare(cost, x_lens, y_lens, begin, end);
  const auto acc_type =
      cost.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  TORCH_CHECK(r.defined() && r.scalar_type() == acc_type && r.is_contiguous() &&
                  r.dim() == 3 && r.size(0) == pr.B && r.size(1) == pr.Tmax &&
                  r.size(2) == pr.W,
              "R must come from forward_cuda(save_r=True)");
  auto aopt = cost.options().dtype(acc_type);
  at::Tensor gv = grad_values.to(aopt).contiguous();
  TORCH_CHECK(gv.dim() == 1 && gv.size(0) == pr.B, "grad_values must have shape [B]");
  at::Tensor grad = at::zeros({pr.B, pr.Tmax, pr.W}, cost.options());
  at::Tensor e_up = at::empty({pr.B, std::max(1, pr.W)}, aopt);
  at::Tensor e_dg = at::empty({pr.B, std::max(1, pr.W)}, aopt);
  if (pr.B == 0) return grad;
  const Launch l = choose_launch(pr.B, pr.Tmax, pr.W, begin.has_value(), rows_per_block);
  auto stream = at::cuda::getCurrentCUDAStream();

  FLASH_SOFTDTW_DISPATCH(cost.scalar_type(), "softdtw_backward", [&] {
    Params<acc_t> p = base_params<acc_t>(cost, pr, gamma, warp);
    p.r = r.data_ptr<acc_t>();
    p.edge_a = e_up.data_ptr<acc_t>();
    p.edge_b = e_dg.data_ptr<acc_t>();
    p.grad_values = gv.data_ptr<acc_t>();
    p.grad_cost = grad.data_ptr();
    if (l.multi_warp)
      softdtw_backward_kernel<scalar_t, acc_t, true><<<l.blocks, l.threads, 0, stream>>>(p);
    else
      softdtw_backward_kernel<scalar_t, acc_t, false><<<l.blocks, l.threads, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return grad;
}

}  // namespace flash_softdtw
