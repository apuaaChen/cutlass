#################################################################################################
#
# Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Summarize the reshape-permute info used to legalize store nodes
"""
from cutlass.backend.evt.ir import DAGIR
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_get_impl import PassGetImpl
from cutlass.backend.evt.passes.pass_post_permute_reshape import PassPostReshapePermute
from typing import Union


class StoreAbv:
    def __init__(self, node_to_store: str, stride: Union[tuple, None]) -> None:
        """
        The abstract value of store is composed of the non-store node to store 
        and the stride in case that different permutations of the same tensor is
        required
        """
        self.node_to_store = node_to_store
        self.stride = stride
    
    def __str__(self) -> str:
        return f"{self.node_to_store}: {self.stride}"


class PassDuplicatedStoreElimination(EVTPassBase):
    dependencies = [
        PassPostReshapePermute,
    ]

    def requires(self) -> None:
        if not hasattr(self.dag_ir, "output2store"):
            self.dag_ir.output2store = {}
        self.workspace = {}
    
    def call(self):
        for node_meta in self.dag_ir.nodes_meta:
            if node_meta.op == "store":
                inputs = self.dag_ir.get_all_inputs(node_meta.name)
                assert len(inputs) == 1
                input = inputs[0]
                input_abv = self.workspace[input]
                self.workspace[node_meta.name] = StoreAbv(input_abv.node_to_store, node_meta.store_tensor.stride)
            else:
                self.workspace[node_meta.name] = StoreAbv(node_meta.name, None)
            pass
        duplicate_dict = {}
        for key, value in self.workspace.items():
            if value.stride is None:
                continue
            value_str = str(value)
            if value_str not in duplicate_dict:
                duplicate_dict[value_str] = [key,]
            else:
                duplicate_dict[value_str].append(key)
        # Update the output2store mapping in dag ir
        for key, value in duplicate_dict.items():
            for v in value:
                self.dag_ir.output2store[v] = value[0]
            if len(value) > 1:
                for v in value[1:]:
                    inputs = self.dag_ir.get_all_inputs(v)
                    assert len(inputs) == 1
                    input = inputs[0]
                    self.dag_ir.replace_all_uses_with(v, input)
