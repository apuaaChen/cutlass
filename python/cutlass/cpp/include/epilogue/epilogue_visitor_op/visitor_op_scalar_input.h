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
  
  \brief A file contains the epilogue visitor Op with Scalar input
*/

#pragma once
#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Epilogue Visitor operator for the following computation:
///
///  ElementInput C <- device memory
///
/// It can only be a leaf node in the epilogue tree
template <
    typename ElementAccumulator_,  ///< Data type of the Accumulator
    typename InputTileIterator_,   ///< Tile iterator type to read the tensor
    typename ElementInput_=typename InputTileIterator_::Element /// data type of input data
>
class VisitorOpScalarInput {
public:
    using ElementInput = ElementInput_;
    using ElementAccumulator = ElementAccumulator_;

    // using InputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    //     typename InputTileIterator_::ThreadMap,
    //     ElementInput
    // >;
    using InputTileIterator = typename InputIteratorType<ElementInput, InputTileIterator_>::type;

    static int const kElementsPerAccess = InputTileIterator::kElementsPerAccess;
    

    using VisitAccessType = Array<ElementInput, kElementsPerAccess>;

    /// Fragment type of accumulator
    using AccumulatorAccessType = Array<ElementAccumulator, kElementsPerAccess>;

    struct SharedStorage {
        CUTLASS_HOST_DEVICE
        SharedStorage() { }
    };

    /// Host-constructable Argument structure
    struct Arguments {
        ElementInput *input_ptr;                 ///< Pointer to the input tensor in device memory
        
        /// Methods
        CUTLASS_HOST_DEVICE
        Arguments(): input_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(
            ElementInput *input_ptr
        ):
            input_ptr(input_ptr)
        { }
    };

    /// Param structure
    struct Params {
        ElementInput *input_ptr;


        /// Method
        CUTLASS_HOST_DEVICE
        Params():
            input_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            input_ptr(args.input_ptr)
        { }
    };

private:
    VisitAccessType scalar_;

public:
    /// Constructs the function object
    CUTLASS_HOST_DEVICE
    VisitorOpScalarInput(
        Params const &params,
        SharedStorage &shared_storage,
        int thread_idx,
        MatrixCoord threadblock_offset,
        MatrixCoord problem_size
    ) {
        ElementInput scalar = *(params.input_ptr);
        scalar_.fill(scalar);
    }
      
    CUTLASS_DEVICE
    void set_batch_index(int batch_idx) {}
    
    CUTLASS_DEVICE
    void begin_epilogue() { }

    CUTLASS_DEVICE
    void begin_step(int step_idx) {}

    CUTLASS_DEVICE
    void begin_row(int row_idx) { }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frag_idx,
        AccumulatorAccessType const &accum
    ) {
        return scalar_;
    }

    CUTLASS_DEVICE
    VisitAccessType visit(
        int row_idx,
        int column_idx,
        int iter_idx,
        AccumulatorAccessType const &accum
    ) {
        return scalar_;
    }

    CUTLASS_DEVICE
    void end_row(int row_idx) { }

    CUTLASS_DEVICE
    void end_step(int step_idx) { }

    CUTLASS_DEVICE
    void end_epilogue() { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////