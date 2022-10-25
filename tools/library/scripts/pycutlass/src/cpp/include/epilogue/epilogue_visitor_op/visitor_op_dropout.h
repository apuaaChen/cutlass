/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * this layernormware without specific prior written permission.
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
  
  \brief A file contains the epilogue visitor Op with dropout
*/

#pragma once
#include "cutlass/cutlass.h"
#include "curand.h"
#include "curand_kernel.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Epilogue Visitor operator for the following computation:
///
///  ElementCompute output;
///  ElementMask    mask;
///  mask, output = dropout(input)
///  mask -> global memory
///
template <
    typename ElementAccumulator_,  ///< Data type of the Accumulator
    typename ElementCompute_,      ///< Data type used to compute dropout
    int      kElementsPerAccess_,  ///< Number of elements computed per operation
    typename ElementMask_,         ///< Data type of the output mask
    typename Visitor_              ///< Child Node
>
class VisitorOpDropoutForward{
public:
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    static int const kElementsPerAccess = kElementsPerAccess_;

    using Visitor = Visitor_;

     /// Fragment type returned from Visitor.visit
    using VisitAccessTypeVisitor = typename Visitor::VisitAccessType;
    using ElementVisit = typename VisitAccessTypeVisitor::Element;

    /// Fragment type returned by this visitor
    using VisitAccessType = Array<ElementCompute, kElementsPerAccess>; 

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    static_assert(kElementsPerAccess==VisitAccessTypeVisitor::kElements, "kElementsPerAccess mismatches with Visitor");

    /// the curand_uniform4 function returns 4 random values per round
    static int const kRandVectorPerVisit = kElementsPerAccess / 4;

    using RandType = Array<float, kElementsPerAccess>;

    /// SMEM buffer class required in the epilogue visitor
    struct SharedStorage {
        typename Visitor::SharedStorage storage_visitor;

        CUTLASS_HOST_DEVICE
        SharedStorage() {}
    };

    /// Host-constructable Arguments structure
    struct Arguments {
        float p;                    ///< Probability of dropout
        uint64_t seed;              ///< random seed
        uint64_t offset;            ///< Absolute offset into subsequence
        typename Visitor::Arguments visitor_arg;    ///< Argument type for visitor

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Arguments() { }

        CUTLASS_HOST_DEVICE
        Arguments(
            float p,
            uint64_t seed,
            uint64_t offset,
            typename Visitor::Arguments visitor_arg
        ):
            p(p), seed(seed), offset(offset), visitor_arg(visitor_arg)
        { }
    };

    /// Parameter structure
    struct Params {
        float p;                    ///< Probability of dropout
        uint64_t seed;              ///< random seed
        uint64_t offset;            ///< Absolute offset into subsequence
        typename Visitor::Params visitor_param;    ///< Argument type for visitor

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Params() { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            p(args.p), seed(args.seed), offset(args.offset),
            visitor_param(args.visitor_arg)
        { }
    };

private:
    curandStatePhilox4_32_10_t state;
    Visitor visitor_;                                  ///< visitor
    int thread_idx_;
    MatrixCoord threadblock_offset;
    MatrixCoord problem_size_;

    RandType rand;

    float p;

    cutlass::multiplies<VisitAccessType> mul_op;

public:
    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpDropoutForward(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        visitor_(params.visitor_param, shared_storage.storage_visitor,
            thread_idx, threadblock_offset, problem_size),
        thread_idx_(thread_idx),
        threadblock_offset(threadblock_offset),
        problem_size_(problem_size),
        p(params.p)
    {
        /// unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t
        curand_init(params.seed, uint64_t(thread_idx), params.offset, &state);
    }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        visitor_.set_batch_index(batch_idx);
    }

    CUTLASS_DEVICE
    void begin_epilogue() {
        visitor_.begin_epilogue();
    }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        visitor_.begin_step(step_idx);
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
        visitor_.begin_row(row_idx);
    }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorAccessType const &accum
    ) {
        /// Get result from visitor
        VisitAccessTypeVisitor result = visitor_.visit(iter_idx, row_idx, column_idx, frag_idx, accum);

        float4* rand_ptr = reinterpret_cast<float4*>(rand.data());

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < kRandVectorPerVisit; ++i) {
            *rand_ptr = curand_uniform4(&state);
            rand_ptr ++;
        }

        VisitAccessType mask;
        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < kElementsPerAccess; ++i) {
            mask[i] = ElementCompute(rand[i] < p);
        }

        /// Type conversion of the dropout mask
        NumericArrayConverter<ElementCompute, ElementVisit, kElementsPerAccess> src_converter;

        VisitAccessType output = mul_op(src_converter(result), mask);

        return output;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {
        visitor_.end_row(row_idx);
    }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        visitor_.end_step(step_idx);
    }

    CUTLASS_DEVICE
    void end_epilogue() {
        visitor_.end_epilogue();
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////