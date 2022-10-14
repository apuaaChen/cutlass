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

    \brief A file contains the epilogue visitor Op with broadcast
*/

#pragma once
#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue Visitor operator for onehot broadcast
template <
    typename ElementAccumulator_,  ///< Data type of the Accumulator
    typename ElementCompute_,      ///< Data type used to compute linear combination
    int      kElementsPerAccess_,  ///< Number of elements computed per operation
    typename Visitor_              ///< Child node
>
class VisitorOpOneHot{
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
    
    /// Fragment type storing the indices
    using IndexFragment = Array<int, kElementsPerAccess>;

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    static_assert(kElementsPerAccess==VisitAccessTypeVisitor::kElements, "kElementsPerAccess mismatches with Visitor");

    /// SMEM buffer class required in the epilogue visitor
    struct SharedStorage {
        typename Visitor::SharedStorage storage_visitor;

        CUTLASS_HOST_DEVICE
        SharedStorage() {}
    };

    /// Host-constructable Arguments structure
    struct Arguments {
        typename Visitor::Arguments visitor_arg;    ///< Argument type for visitor

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Arguments() { }
        
        CUTLASS_HOST_DEVICE
        Arguments(
            typename Visitor::Arguments visitor_arg
        ):
            visitor_arg(visitor_arg)
        { }
    };

    /// Parameter structure
    struct Params {
        typename Visitor::Params visitor_param;    ///< Argument type for visitor

        //
        // Methods
        //
        CUTLASS_HOST_DEVICE
        Params() { }
        
        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            visitor_param(args.visitor_arg)
        { }
    };

private:
    //
    // Data members
    //

    Visitor visitor_op;

public:

    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpOneHot(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ):
        visitor_op(params.visitor_param, shared_storage.storage_visitor, thread_idx, threadblock_offset, problem_size)
    { }

    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {
        visitor_op.set_batch_index(batch_idx);
    }

    CUTLASS_DEVICE
    void begin_epilogue() {
        visitor_op.begin_epilogue();
    }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {
        visitor_op.begin_step(step_idx);
    }

    CUTLASS_DEVICE
    void begin_row(int row_idx) {
        visitor_op.begin_row(row_idx);
    }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorAccessType const &accum
    ) { 
        /// Get result from visitor A and visitor B
        VisitAccessTypeVisitor result;
        result = visitor_op.visit(iter_idx, row_idx, column_idx, frag_idx, accum);

        /// Type conversion to column index
        NumericArrayConverter<int, ElementVisit, kElementsPerAccess> index_converter;
        
        IndexFragment target_indices = index_converter(result);

        VisitAccessType one_hot;

        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++){
            *(one_hot.data() + i) = ElementCompute(*(target_indices.data() + i) == column_idx + i);
        }

        return one_hot;
    }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int row_idx,
        int column_idx,
        AccumulatorAccessType const &accum
    ) { 
        /// Get result from visitor A and visitor B
        VisitAccessTypeVisitor result;
        result = visitor_op.visit(row_idx, column_idx, accum);

        /// Type conversion to column index
        NumericArrayConverter<int, ElementVisit, kElementsPerAccess> index_converter;
        
        IndexFragment target_indices = index_converter(result);

        VisitAccessType one_hot;

        #pragma unroll
        for (int i = 0; i < kElementsPerAccess; i ++){
            *(one_hot.data() + i) = ElementCompute(*(target_indices.data() + i) == column_idx + i);
        }

        return one_hot;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) {
        visitor_op.end_row(row_idx);
    }

    CUTLASS_DEVICE
    void end_step(int step_idx) {
        visitor_op.end_step(step_idx);
    }

    CUTLASS_DEVICE
    void end_epilogue() {
        visitor_op.end_epilogue();
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////