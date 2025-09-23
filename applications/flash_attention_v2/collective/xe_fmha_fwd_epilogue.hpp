/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"


namespace cutlass::fmha::collective {

using namespace cute;

template <class TiledMMAPV_,        // Tiling for P*V GEMM
          class TileShapeO_,        // Shape of output tile, may be larger than P*V GEMM
          class TensorO_,           // 2D slice of global output tensor
          class TiledCopyO_ = void> // Optional TiledCopy for loading O
class FMHAFwdEpilogue {

public:
  //
  // Type Aliases
  //
  using TiledMMAPV = TiledMMAPV_;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(take<1,4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D = decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_,_),0)));
  using ElementO = typename TensorO_::value_type;

  // Split k-reduced tiles between participating subgroups.
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

//  using SG_O_M = Int<cute::gcd(get<0>(TileShapePV{}), SGPerWG{})>;
//  using SG_O_N = Int<cute::min(get<1>(TileShapePV{}) / intel::_SGSize{}, SGPerWG{} / ThreadsO_M{})>;
//  using SG_Out = decltype(SG_O_M{} * SG_O_N{});
//
//  using SGTileShapeOut = decltype(shape_div(take<0,2>(SGTileShapePV{}), Shape<SG_O_M, SG_O_N>{}));

  using DefaultTiledCopyO = conditional_t<ReduceK{} == _1{},
                                          decltype(make_block_2d_copy_C(TiledMMAPV{}, TensorO2D{})),
                                          decltype(make_block_2d_copy(XE_STORE_2D<sizeof_bits_v<ElementO>, 1, 16>{}, TensorO2D{}))>;
  using TiledCopyO = conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  // Stateless design -- no arguments or parameters.
  struct Arguments {};
  struct Params {};

  struct SharedStorage {
    // TODO reduce-k buffers
  };

  static constexpr
  Params to_underlying_arguments(Arguments const &args, void * /* workspace */) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FMHAFwdEpilogue(Params const&, SharedStorage&) {}

  template <typename FragA, typename FragARow, typename QVCoord>
  CUTLASS_DEVICE
  void
  operator()(TensorO2D const& O,        // Global O tensor: (q,v)
             FragA          & tArA,     // O accumulator:   (q,v)
             FragARow       & tA_max,   // Softmax row-wise max accumulator
             FragARow       & tA_sum,   // Softmax row-wise sum accumulator
             QVCoord          blk_qv,   // WG tile indices: (q,v)
             int              thr_id) { // Work-item ID

    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_sum] = reduce_A(tArA, tA_max, tA_sum);

    /* Complete softmax, dividing out sums. */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      rA_sum(i) = ElementA(1) / rA_sum(i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape());          // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);     // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  // Note that each k block has its own scale factor based on O_max,
  //   so A/A_sum contributions need to be rescaled to match.
  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE
  decltype(auto)
  reduce_A(FragA        & tArA,     // O accumulator:   (q,v)
           FragARow     & tA_max,   // Softmax row-wise max accumulator
           FragARow     & tA_sum) { // Softmax row-wise sum accumulator

    if constexpr (ReduceK{} == _1{}) {
      return std::tie(tArA, tA_sum);
    } else {
      // implementme
      // sycl::group_barrier(group);
    }
  }
};


} // namespace cutlass::fmha::collective
