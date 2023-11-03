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
Sm80 RowMajor Output pass
"""
from cutlass.backend.evt.ir.node import NodeBase
from cutlass.backend.evt.ir.store_nodes import StoreNode, AuxStoreImpl
from cutlass.backend.evt.ir.layout_nodes import LayoutNode
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_layout_elimination import PassLayoutManipulateElimination
from cutlass.backend import LayoutType
from cutlass.epilogue.evt_ops import permute


class PassSm80RowMajorOutputPass(EVTPassBase):
    """
    Sm80 kernels require the aux store nodes to be in row major. Which means
    the last dim of their store_stride should be 1. However, sometimes the 
    permutation in the kernel may lead to column major output. 
    """
    dependencies = [PassLayoutManipulateElimination]

    def call(self) -> None:
        # Sm90 kernel supports ColumnMajor output
        if self.cc > 80:
            return

        # Get the problem shape
        accumulator = self.dag_ir.get_node_meta("accum")
        problem_size = accumulator.tensor.shape

        
        # Find the major of all store nodes in the IR
        output_layouts = set()
        for node in self.dag_ir.nodes:
            node_meta: NodeBase = self.dag_ir.get_node_meta(node)
            if node_meta.op != "store":
                continue
            if not AuxStoreImpl.match(node_meta, problem_size):
                continue
            store_tensor = node_meta.store_tensor
            store_stride = store_tensor.stride
            if store_stride[-1] == 1:
                output_layouts.add(LayoutType.RowMajor)
            else:
                output_layouts.add(LayoutType.ColumnMajor)
        
        if len(output_layouts) != 1:
            raise NotImplementedError("Got both RowMajor and ColumnMajor outputs")
        
        layout = list(output_layouts)[0]
        if layout == LayoutType.RowMajor:
            return
        
        # Step 1: Permute the tensor of the accumulator node
        accumulator.tensor.permute((0, 2, 1))
        # Step 2: Insert the permutation node right behind the accumulator
        name = "accum_transpose"
        layout_node = LayoutNode(name=name, fn=permute, kwargs={"indices":(0, 2, 1)})
        self.dag_ir.add_node(layout_node)

        # Step 3: Set up edges
        # 3.1 Find all the users of accumulator
        users = self.dag_ir.get_users("accum")
        # 3.2 Remove the edges between accum and its users
        edge_weights = []
        for user in users:
            edge_weights.append(self.dag_ir.get_edge_weight("accum", user))
            self.dag_ir.remove_edge("accum", user)
        # 3.3 Add edge accum -> layout
        self.dag_ir.add_edge("accum", name)
        # 3.4 Add edge layout -> user
        for user, weight in zip(users, edge_weights):
            self.dag_ir.add_edge(name, user, weight)
        
        # Step 4: rerun the shape type propagation on the current layout node
        layout_node.type_propagation([accumulator])
        layout_node.shape_propagation([accumulator])

        # Step 5: redo the layout elimination pass
        layout_eliminate_pass = PassLayoutManipulateElimination(self.dag_ir)
        layout_eliminate_pass()

        # Step 6: register the switched info to the dag ir
        self.dag_ir.transposed = True

