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
  typename ConvProblemSize = Conv2dProblemSize   ///! Convolutional operator on 2D or 3D problem
>
struct ImplicitGemmConvolutionStridedDgradWithVisitor:
  ImplicitGemmConvolutionStridedDgrad<
    Mma, Epilogue, ThreadblockSwizzle_, ConvOperator, ConvProblemSize> {
public:
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  
  using Base = ImplicitGemmConvolutionStridedDgrad<
    Mma, Epilogue, ThreadblockSwizzle_, ConvOperator, ConvProblemSize>;
  using Base::Base;

  using FusionCallbacks = typename Epilogue::FusionCallbacks;
  using ThreadblockShape = typename Base::ThreadblockShape;
  static Operator const kConvolutionalOperator = ConvOperator;
  using Arguments = typename Base::Arguments;

  /// Parameters structure
  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    cute::Shape<int32_t,int32_t,int32_t> implicit_gemm_problem_shape;
    int swizzle_log_tile;
    FastDivmod stride_h_divmod;
    FastDivmod stride_w_divmod;
    int gemm_k_iterations;
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
    Params(): gemm_k_iterations(0) { }

    /// 
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      stride_h_divmod(args.problem_size.stride_h),
      stride_w_divmod(args.problem_size.stride_w),
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
        kConvolutionalOperator, ThreadblockShape::kK, args.problem_size);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
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
  ImplicitGemmConvolutionStridedDgradWithVisitor() { } 

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

    // Compute starting filter position for strided dgrad
    int tile_m_per_filter = strided_dgrad_tile_m_per_filter(params.problem_size, 
                                                            ThreadblockShape::kM);
    int filter_tile_m = (threadblock_tile_idx.m() / tile_m_per_filter);
    

    // The subsequent fast_divmod() operations are equivalent to the following logical computation:
    //
    // int start_r = filter_tile_m / (params.problem_size.stride_w);
    // int start_s = filter_tile_m % (params.problem_size.stride_w);

    int start_r, start_s;
    params.stride_w_divmod(start_r, start_s, filter_tile_m);

    int filter_r = start_r;
    int filter_s = start_s;

    if (params.problem_size.mode == Mode::kConvolution) {
      filter_r = (params.problem_size.R - 1 - filter_r);
      filter_s = (params.problem_size.S - 1 - filter_s);
    }

    // Starting h, w positions for filter position in gemm_k=0
    int start_h, start_w;
    strided_dgrad_starting_coords(
      params.problem_size,
      params.stride_h_divmod, params.stride_w_divmod,
      filter_r, filter_s,
      start_h, start_w);

    if (start_h >= params.problem_size.H || start_w >= params.problem_size.W) {
      return;
    }

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    // Check if CTA contributes valid MMA (Dy * w) and accumulator will be non-zero after MMA
    if (start_r < params.problem_size.R && start_s < params.problem_size.S) {
      // Scale gemm_k_iterations for strided dgrad
      int gemm_k_iterations = (params.gemm_k_iterations / (params.problem_size.R * params.problem_size.S)
                              ) * params.problem_size.num_gemm_k_filter_positions(start_r, start_s);
      
      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params.iterator_A,
        params.problem_size,
        params.ptr_A,
        thread_idx,
        params.stride_h_divmod, params.stride_w_divmod,
        start_r, start_s,
        MatrixCoord(
          threadblock_tile_idx.m() * Mma::Shape::kM,
          threadblock_tile_idx.k() * Mma::Shape::kK
        ) 
      );
      
      typename Mma::IteratorB iterator_B(
        params.iterator_B,
        params.problem_size,
        params.ptr_B,
        thread_idx,
        start_r, start_s,
        MatrixCoord(
          threadblock_tile_idx.k() * Mma::Shape::kK,
          threadblock_tile_idx.n() * Mma::Shape::kN
        )
      );

      //
      // Main loop
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

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
      lane_idx
    );

    // Compute the row mapping params

    int start_h_, start_w_;
    strided_dgrad_starting_coords(
      params.problem_size,
      params.stride_h_divmod, params.stride_w_divmod,
      start_r, start_s,
      start_h_, start_w_
    );
    int tiled_rows_per_filter = tile_m_per_filter * ThreadblockShape::kM;

    epilogue(accumulators, threadblock_tile_idx, params.implicit_gemm_problem_shape, thread_idx, params.problem_size, start_h_, start_w_, tiled_rows_per_filter);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
