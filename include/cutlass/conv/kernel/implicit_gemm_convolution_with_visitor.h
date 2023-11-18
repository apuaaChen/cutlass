/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
    \brief Template for a pipelined Implicit GEMM kernel with an epilogue defined under the epilogue visitor concept
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename Mma,                                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad)
  typename ConvProblemSize = Conv2dProblemSize,  ///! Convolutional operator on 2D or 3D problem
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone    ///! Group mode
>
struct ImplicitGemmConvolutionWithVisitor:
  ImplicitGemmConvolution<Mma, Epilogue, ThreadblockSwizzle_, ConvOperator, ConvProblemSize, GroupMode_>{
public:
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using Base = ImplicitGemmConvolution<
    Mma, Epilogue, ThreadblockSwizzle_, 
    ConvOperator, ConvProblemSize, GroupMode_>;
  using Base::Base;

  static conv::GroupMode const kGroupMode = Base::kGroupMode;
  static Operator const kConvolutionalOperator = Base::kConvolutionalOperator;
  static IteratorAlgorithm const kIteratorAlgorithm = Base::kIteratorAlgorithm;

  using ThreadblockShape = typename Base::ThreadblockShape;
  using FusionCallbacks = typename Epilogue::FusionCallbacks;

  using Arguments = typename Base::Arguments;
  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    cute::Shape<int32_t,int32_t,int32_t> implicit_gemm_problem_shape;
    int swizzle_log_tile;
    int gemm_k_iterations;
    int gemm_k_iterations_per_channel;
    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const *ptr_A;
    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const *ptr_B;
    typename FusionCallbacks::Params output_op;
    int *semaphore;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), gemm_k_iterations(0) { }

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      implicit_gemm_problem_shape({implicit_gemm_problem_size.m(), implicit_gemm_problem_size.n(), 1}),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      output_op(FusionCallbacks::to_underlying_arguments(implicit_gemm_problem_shape, args.output_op, nullptr /*workspace*/)),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(
        kConvolutionalOperator,
        ThreadblockShape::kK,
        args.problem_size,
        kIteratorAlgorithm,
        kGroupMode,
        ThreadblockShape::kN);

      gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
          kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        implicit_gemm_problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  using SharedStorage = typename Base::SharedStorage;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolutionWithVisitor() { }

  /// Executes one ImplicitGEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;
    if (kGroupMode != GroupMode::kNone) {
      if (kGroupMode != GroupMode::kDepthwise) {
        int k_per_group = params.problem_size.K / params.problem_size.groups;
        int group_idx = threadblock_tile_idx.n() * Mma::Shape::kN / k_per_group;
        int channels_per_group = params.problem_size.C / params.problem_size.groups;
        iterator_A_column_offset += group_idx * channels_per_group;
      } else {
        iterator_A_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
      }
    } 

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        iterator_A_column_offset
      )
    );
    
    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_B,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      )
    );

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, params.gemm_k_iterations_per_channel);

    //
    // Epilogue
    //

    // Compute logical position within grid
    threadblock_tile_idx =
      threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
    
    Epilogue epilogue(
      params.output_op,
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    epilogue(accumulators, threadblock_tile_idx, params.implicit_gemm_problem_shape, thread_idx);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////