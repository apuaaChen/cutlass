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
Assign each node a legal name to improve the cache hit rate of the same graph
"""
from cutlass.backend.evt.passes.pass_manager import EVTPassBase
from cutlass.backend.evt.passes.pass_dag_2_tree import PassDAG2Tree
from cutlass.backend.evt.ir import DAGIR

class PassAssignLegalName(EVTPassBase):
    dependencies = [
        PassDAG2Tree
    ]

    def call(self):
        self.visited = []
        self.load_cnt = 0
        self.compute_cnt = 0
        self.store_cnt = 0
        self.dag_cnt = 0


        # Start from accumulator
        accum_node = None
        if self.dag_ir.has_node("accum_t"):
            accum_node = "accum_t"
        else:
            assert self.dag_ir.has_node("accum")
            accum_node = "accum"
        
        self.visit(accum_node, self.dag_ir)
    
    def visit(self, node: str, graph_ir: DAGIR):
        if node in self.visited:
            return
        node_meta = graph_ir.get_node_meta(node)
        if node_meta.disabled:
            return
        # visit the current node
        self.visited.append(node)
        # Update the legal name of the current node
        op = node_meta.op
        cnt = getattr(self, f"{op}_cnt")
        legal_name = f"{op}_{cnt}"
        setattr(self, f"{op}_cnt", cnt +1)
        node_meta.underlying_impl.legal_name = legal_name

        # Special rule for dag
        if op == "dag":
            self.visit(node_meta.output_node.name, node_meta.subgraph)
            
        # Visit all its incoming edges sorted by edge weight
        inputs = graph_ir.get_all_inputs(node)
        for input in inputs:
            self.visit(input, graph_ir)
        # Visit all the out going edges (TODO: may cause indeterministic result)
        users = graph_ir.get_users(node)
        for user in users:
            self.visit(user, graph_ir)
