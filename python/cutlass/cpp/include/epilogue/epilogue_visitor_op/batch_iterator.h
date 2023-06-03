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
  
  \brief A file contains the epilogue visitor Op with broadcasting vector to all columns
*/

#pragma once
#include "cutlass/cutlass.h"
#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

class EpilogueVisitorBatchIterator {
public:
    /// Host-constructable argument structure
    struct Arguments {
        int64_t stride;
        float factor;  // the factor could be smaller than 0. We store it with floating point data type
        int modulo;

        /// Methods
        CUTLASS_HOST_DEVICE
        Arguments() { }

        CUTLASS_HOST_DEVICE
        Arguments(int64_t stride, float factor, int modulo):
            stride(stride), factor(factor), modulo(modulo) { }
    };

    /// Param structure
    struct Params {
        int64_t stride;
        float factor;  // the factor could be smaller than 0. We store it with floating point data type
        int modulo;

        /// Methods
        CUTLASS_HOST_DEVICE
        Params() { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args):
            stride(args.stride), factor(args.factor), modulo(args.modulo)
        { }
    };
private:

    int64_t stride_;
    float factor_;
    int modulo_;

public:

    /// Constructor
    CUTLASS_HOST_DEVICE
    EpilogueVisitorBatchIterator(
        Params const &params
    ):
        stride_(params.stride),
        factor_(params.factor),
        modulo_(params.modulo)
    { }

    CUTLASS_DEVICE
    int64_t batch_offset(int batch_idx) {
        return stride_ * (int(float(batch_idx) * factor_) % modulo_);
    }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////