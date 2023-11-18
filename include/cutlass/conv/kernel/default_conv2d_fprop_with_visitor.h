/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief 
    Default kernel-level implicit GEMM convolution definitions combine threadblock-scoped 
      matrix multiply-add with fused epilogue visitor callbacks
*/

#pragma once
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution_with_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ElementEpilogue,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename FusionCallbacks,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
  conv::StrideSupport StrideSupport = StrideSupport::kStrided,
  /// Access granularity of A matrix in units of elements
  int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value,
  /// Access granularity of C matrix in units of elements
  int AlignmentC = 128 / cutlass::sizeof_bits<ElementB>::value,
  /// Number of stages used in the pipelined epilogue,
  int EpilogueStages = 1
> struct DefaultConv2dFpropWithVisitor {
    using Base = DefaultConv2dFprop<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
      WarpShape, InstructionShape, 
      epilogue::thread::LinearCombination<
        ElementC, AlignmentC, ElementAccumulator, ElementEpilogue
      >,
      ThreadblockSwizzle, Stages, MathOperatorTag, IteratorAlgorithm,
      StrideSupport, AlignmentA, AlignmentB>;

    using Mma = typename Base::Mma;
    static const int kPartitionsK = Base::kPartitionsK;

    // Define epilogue
    using Epilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
      typename Base::Epilogue,
      FusionCallbacks,
      EpilogueStages
    >;

    // Define the kernel
    using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionWithVisitor<
      Mma,
      Epilogue,
      ThreadblockSwizzle,
      conv::Operator::kFprop
    >;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
