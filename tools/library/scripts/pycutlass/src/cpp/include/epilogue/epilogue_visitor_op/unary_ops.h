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
  
  \brief A file contains the unary ops
*/

#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Scalar multiplication
/// The TPtr is used to handle cases where the pointer has different data type
template <typename T, int N, typename TPtr=T>
struct Mult {

    struct Arguments {
        T alpha;
        TPtr* alpha_ptr;

        CUTLASS_HOST_DEVICE
        Arguments(): alpha(T(1.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(T alpha): alpha(alpha), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(TPtr* alpha_ptr): alpha(T(0.0)), alpha_ptr(alpha_ptr) { }
    };

    struct Params {
        T alpha;      ///< scalar adder
        TPtr* alpha_ptr; ///< fetch scalar from device

        CUTLASS_HOST_DEVICE
        Params(): alpha(T(1.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): 
            alpha(args.alpha), alpha_ptr(args.alpha_ptr) { }
    };

    T alpha_;

    CUTLASS_HOST_DEVICE
    Mult(
        Params const &params
    ) {
        if (params.alpha_ptr) alpha_ = T(*(params.alpha_ptr));
        else alpha_ = params.alpha;
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &source) const {
        cutlass::multiplies<Array<T, N>> multiply_op;
        return multiply_op(source, alpha_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return alpha_ != T(0);
    }

};


/// Scalar multiplication
/// The TPtr is used to handle cases where the pointer has different data type
template <typename T, int N, typename TPtr=T>
struct Div {

    struct Arguments {
        T alpha;
        TPtr* alpha_ptr;

        CUTLASS_HOST_DEVICE
        Arguments(): alpha(T(1.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(T alpha): alpha(alpha), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(TPtr* alpha_ptr): alpha(T(0.0)), alpha_ptr(alpha_ptr) { }
    };

    struct Params {
        T alpha;      ///< scalar adder
        TPtr* alpha_ptr; ///< fetch scalar from device

        CUTLASS_HOST_DEVICE
        Params(): alpha(T(1.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): 
            alpha(args.alpha), alpha_ptr(args.alpha_ptr) { }
    };

    T alpha_;

    CUTLASS_HOST_DEVICE
    Div(
        Params const &params
    ) {
        if (params.alpha_ptr) alpha_ = T(*(params.alpha_ptr));
        else alpha_ = params.alpha;
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &source) const {
        cutlass::divides<Array<T, N>> div_op;
        return div_op(source, alpha_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }

};


/// Scalar addition
template <typename T, int N, typename TPtr=T>
struct Add {

    struct Arguments {
        T alpha;
        TPtr* alpha_ptr;

        CUTLASS_HOST_DEVICE
        Arguments(): alpha(T(0.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(T alpha): alpha(alpha), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(TPtr* alpha_ptr): alpha(T(0.0)), alpha_ptr(alpha_ptr) { }
    };

    struct Params {
        T alpha;      ///< scalar adder
        TPtr* alpha_ptr; ///< fetch scalar from device

        CUTLASS_HOST_DEVICE
        Params(): alpha(T(0.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): 
            alpha(args.alpha), alpha_ptr(args.alpha_ptr) { }
    };

    T alpha_;
    CUTLASS_HOST_DEVICE
    Add(
        Params const &params
    ) {
        if (params.alpha_ptr) alpha_ = T(*(params.alpha_ptr));
        else alpha_ = params.alpha;
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &source) const {
        cutlass::plus<Array<T, N>> plus_op;
        return plus_op(source, alpha_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};


/// Scalar Sub
template <typename T, int N, typename TPtr=T>
struct Sub {

    struct Arguments {
        T alpha;
        TPtr* alpha_ptr;

        CUTLASS_HOST_DEVICE
        Arguments(): alpha(T(0.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(T alpha): alpha(alpha), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(TPtr* alpha_ptr): alpha(T(0.0)), alpha_ptr(alpha_ptr) { }
    };

    struct Params {
        T alpha;      ///< scalar adder
        TPtr* alpha_ptr; ///< fetch scalar from device

        CUTLASS_HOST_DEVICE
        Params(): alpha(T(0.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): 
            alpha(args.alpha), alpha_ptr(args.alpha_ptr) { }
    };

    T alpha_;
    CUTLASS_HOST_DEVICE
    Sub(
        Params const &params
    ) {
        if (params.alpha_ptr) alpha_ = T(*(params.alpha_ptr));
        else alpha_ = params.alpha;
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &source) const {
        cutlass::minus<Array<T, N>> sub_op;
        return sub_op(source, alpha_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};


/// Scalar addition
template <typename T, int N, typename TPtr=T>
struct Ne {

    struct Arguments {
        T alpha;
        TPtr* alpha_ptr;

        CUTLASS_HOST_DEVICE
        Arguments(): alpha(T(0.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(T alpha): alpha(alpha), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Arguments(TPtr* alpha_ptr): alpha(T(0.0)), alpha_ptr(alpha_ptr) { }
    };

    struct Params {
        T alpha;      ///< scalar adder
        TPtr* alpha_ptr; ///< fetch scalar from device

        CUTLASS_HOST_DEVICE
        Params(): alpha(T(0.0)), alpha_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): 
            alpha(args.alpha), alpha_ptr(args.alpha_ptr) { }
    };

    T alpha_;
    CUTLASS_HOST_DEVICE
    Ne(
        Params const &params
    ) {
        if (params.alpha_ptr) alpha_ = T(*(params.alpha_ptr));
        else alpha_ = params.alpha;
    }

    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &source) const {
        Array<T, N> result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i ++) {
            result[i] = T(source[i] != alpha_);
        }
        return result;
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};


/// ReLU
template <typename T, int N, typename TPtr=T>
struct ReLUVisitor {
    struct Arguments {
        T threshold;

        CUTLASS_HOST_DEVICE
        Arguments():threshold(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Arguments(T threshold): threshold(threshold) { }
    };

    struct Params {
        T threshold;

        CUTLASS_HOST_DEVICE
        Params():threshold(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): threshold(args.threshold) { }
    };

    T threshold_;

    CUTLASS_HOST_DEVICE
    ReLUVisitor(Params const &params):
        threshold_(params.threshold) { }
    
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        maximum<Array<T, N>> mx;
        return mx(frag, threshold_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};

/// leakyReLU
template <typename T, int N, typename TPtr=T>
struct LeakyReLUVisitor {
    struct Arguments {
        T leaky_alpha;

        CUTLASS_HOST_DEVICE
        Arguments():leaky_alpha(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Arguments(T leaky_alpha): leaky_alpha(leaky_alpha) { }
    };

    struct Params {
        T leaky_alpha;

        CUTLASS_HOST_DEVICE
        Params():leaky_alpha(T(0.0)) { }

        CUTLASS_HOST_DEVICE
        Params(Arguments const &args): leaky_alpha(args.leaky_alpha) { }
    };

    T leaky_alpha_;

    CUTLASS_HOST_DEVICE
    LeakyReLUVisitor(Params const &params):
        leaky_alpha_(params.leaky_alpha) { }
    
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        cutlass::epilogue::thread::LeakyReLU<Array<T, N>> leaky_op;
        return leaky_op(frag, leaky_alpha_);
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
    
};

/// Tanh
template <typename T, int N, typename TPtr=T>
struct TanhVisitor {
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
    TanhVisitor(Params const &params) { }

    // scalar operator
    CUTLASS_HOST_DEVICE
    T tanh_op(T const &scalar) const {
        return fast_tanh(scalar);
    }

    /// vector operator
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        Array<T, N> y;

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < N; ++i) {
            y[i] = tanh_op(frag[i]);
        }

        return y;
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};


/// GeLU: tanh approximation
// https://github.com/pytorch/pytorch/blob/d321be61c07bc1201c7fe10cd03d045277a326c1/aten/src/ATen/native/cuda/ActivationGeluKernel.cu
template <typename T, int N, typename TPtr=T>
struct GeluForwardVisitor {

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
    GeluForwardVisitor(Params const &params) { }

    // scalar operator
    CUTLASS_HOST_DEVICE
    T gelu_op(T const &scalar) const {
        T x_cube = scalar * scalar * scalar;
        T inner = kBeta * (scalar + kKappa * x_cube);
        return T(0.5) * scalar * (T(1.0) + fast_tanh(inner));
    }

    /// vector operator
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        Array<T, N> y;

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < N; ++i) {
            y[i] = gelu_op(frag[i]);
        }

        return y;
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};


/// Sigmoid
template <typename T, int N, typename TPtr=T>
struct SigmoidVisitor {
    /// Arguments
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
    SigmoidVisitor(Params const &params) { }

    /// Scalar operator
    CUTLASS_HOST_DEVICE
    T sigmoid_op(T const &scalar) const {
        return T(1) / (T(1) + fast_exp(-scalar));
    }

    /// vector operator
    CUTLASS_HOST_DEVICE
    Array<T, N> operator()(Array<T, N> const &frag) const {
        Array<T, N> y;

        CUTLASS_PRAGMA_UNROLL
        for (int i=0; i < N; ++i) {
            y[i] = sigmoid_op(frag[i]);
        }

        return y;
    }

    CUTLASS_HOST_DEVICE
    bool guard() {
        return true;
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
