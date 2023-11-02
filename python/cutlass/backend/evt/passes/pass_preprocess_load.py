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
Preprocess the load nodes

Some of the nodes like rand-like and one-hot are parsed as compute node. This 
pass fixes them and converts them back to a single load node
"""

from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_shape_type_propagation import PassShapeTypePropagation
from cutlass.backend.evt.ir import ComputeNode, LoadNode
from cutlass.backend.library import FunctionalOp
from copy import deepcopy

class PassPreprocessLoad(EVTPassBase):
    """
    Preprocess load nodes
    """
    dependencies = [PassShapeTypePropagation]

    def call(self):
        # Step 1: find the compute nodes with op == "Rand"
        rand_nodes = []
        for node_meta in self.dag_ir.nodes_meta:
            if isinstance(node_meta, ComputeNode):
                if node_meta.fn == FunctionalOp.Rand:
                    rand_nodes.append(node_meta.name)
        
        # Step 2: for each compute, replacing it with an input node
        for node in rand_nodes:
            # insert a load node
            name = f"{node}_load"
            load_node = LoadNode(name)
            load_node.tensor = {
                "tensor": self.dag_ir.get_node_meta(node).tensor
            }
            setattr(load_node, "fn", FunctionalOp.Rand)
            load_node.type_propagation()
            self.dag_ir.add_node(load_node)
            self.dag_ir.replace_all_uses_with(node, name)
