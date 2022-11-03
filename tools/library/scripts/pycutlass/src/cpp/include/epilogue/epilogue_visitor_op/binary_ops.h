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
  
  \brief A file contains the binary ops
*/

#pragma once
#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////


/// vector addition
template <typename T, int N>
struct VectorAdd {

    struct Arguments {
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments():tmp(0){ }

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { }
    };
    
    struct Params {

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args) { }
    };

    CUTLASS_HOST_DEVICE
    VectorAdd(
        Params const &params
    ) { }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
        cutlass::plus<Array<T, N>> add_op;
        return add_op(lhs, rhs);
    }

};


/// vector subtraction
template <typename T, int N>
struct VectorSub {

    struct Arguments {
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments():tmp(0){ }

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { }
    };
    
    struct Params {

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args) { }
    };

    CUTLASS_HOST_DEVICE
    VectorSub(
        Params const &params
    ) { }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
        cutlass::minus<Array<T, N>> minus_op;
        return minus_op(lhs, rhs);
    }

};


/// vector multiplication
template <typename T, int N>
struct VectorMult {

    struct Arguments {
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments():tmp(0){ }

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { }
    };
    
    struct Params {

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args) { }
    };

    CUTLASS_HOST_DEVICE
    VectorMult(
        Params const &params
    ) { }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
        cutlass::multiplies<Array<T, N>> mult_op;
        return mult_op(lhs, rhs);
    }

};


/// vector divide
template <typename T, int N>
struct VectorDiv {

    struct Arguments {
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments():tmp(0){ }

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { }
    };
    
    struct Params {

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args) { }
    };

    CUTLASS_HOST_DEVICE
    VectorDiv(
        Params const &params
    ) { }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
        cutlass::divides<Array<T, N>> div_op;
        return div_op(lhs, rhs);
    }

};


/// GeLU: tanh approximation
// https://github.com/pytorch/pytorch/blob/d321be61c07bc1201c7fe10cd03d045277a326c1/aten/src/ATen/native/cuda/ActivationGeluKernel.cu
template <typename T, int N>
struct VectorGeluBackward {

    static const T _M_SQRT2 = 1.41421356237309504880;
    static const T _M_2_SQRTPI = 1.12837916709551257390;

    static const T kBeta = _M_SQRT2 * _M_2_SQRTPI * T(0.5);
    static const T kKappa = 0.044715;
    /// Argument
    struct Arguments {
        // a placeholder argument to ensure correctness of ctypes
        int tmp;

        CUTLASS_HOST_DEVICE
        Arguments(): tmp(0) { };

        CUTLASS_HOST_DEVICE
        Arguments(int tmp): tmp(tmp) { };
    };

    /// Param
    struct Params {
        CUTLASS_HOST_DEVICE
        Params(){ };
        Params(Arguments const &args) { }
    };

    /// Constructor
    CUTLASS_HOST_DEVICE
    VectorGeluBackward(Params const &params) { }

    // scalar operator
    CUTLASS_HOST_DEVICE
    T gelu_backward_op(T const &dy, T const &x) const {
        // auto x_sq = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
        T x_sq = x * x;
        // auto x_cube = x_sq * static_cast<opmath_t>(x);
        T x_cube = x_sq * x;
        // auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
        T inner = kBeta * (x + kKappa * x_cube);
        // auto tanh_inner = c10::cuda::compat::tanh(inner);
        T tanh_inner = fast_tanh(inner);

        // auto left = opmath_t(0.5) * static_cast<opmath_t>(x);
        T left = T(0.5) * x;
        // auto right = opmath_t(1) + tanh_inner;
        T right = T(1.0) + tanh_inner;

        // auto left_derivative = 0.5 * right;
        T left_derivative = T(0.5) * right;

        // auto tanh_derivative = opmath_t(1) - tanh_inner * tanh_inner;
        T tanh_derivative = T(1.0) - tanh_inner * tanh_inner;
        // auto inner_derivative = kBeta * (opmath_t(1) + opmath_t(3) * kKappa * x_sq);
        T inner_derivative = kBeta * (T(1.0) + T(3.0) * kKappa * x_sq);
        // auto right_derivative = left * tanh_derivative * inner_derivative;
        T right_derivative = left * tanh_derivative * inner_derivative;

        // return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);
        return dy * (left_derivative + right_derivative);
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
        Array<T, N> grad_out;

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < N; ++i) {
            grad_out[i] = gelu_backward_op(lhs[i], rhs[i]);
        }

        return grad_out;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
