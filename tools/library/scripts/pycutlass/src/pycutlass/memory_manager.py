#################################################################################################
#
# Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import rmm
import numpy as np
from cuda import cuda
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False


class PoolMemoryManager:
    """
    Using rmm to manage memory pool
    """
    def __init__(self, init_pool_size: int, max_pool_size: int) -> None:
        self.pool = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=init_pool_size,
            maximum_pool_size=max_pool_size
        )
        self.mr = rmm.mr.TrackingResourceAdaptor(self.pool)
        rmm.mr.set_current_device_resource(self.mr)

    def pool_size(self):
        return self.pool.pool_size()
    
    def todevice(self, host_data, dtype=np.float32):
        """
        Pass the host_data to device memory
        """
        if isinstance(host_data, list):
            return rmm.DeviceBuffer.to_device(np.array(host_data, dtype=dtype).tobytes())
        elif isinstance(host_data, np.ndarray):
            return rmm.DeviceBuffer.to_device(host_data.tobytes())

    def device_mem_alloc(self, size):
        return rmm.DeviceBuffer(size=size)

    def get_allocated_size(self):
        device_resource = rmm.mr.get_current_device_resource()
        return device_resource.get_allocated_bytes()


class TorchDeviceBuffer:
    def __init__(self, size) -> None:
        if isinstance(size, int):
            size = (size,)
        self.tensor = torch.empty(size=size, dtype=torch.int8, device="cuda")
        self.ptr = cuda.CUdeviceptr(self.tensor.data_ptr())

class TorchMemoryManager:
    """
    Using torch to manage memory pool
    """
    def __init__(self) -> None:
        assert torch_available
        self.reserved_buffers = {}

    def todevice(self, host_data, dtype=np.float32):
        if isinstance(host_data, list):
            return torch.from_numpy(np.array(host_data, dtype=dtype)).to("cuda")
        elif isinstance(host_data, np.ndarray):
            return torch.from_numpy(host_data).to("cuda")
        else:
            raise NotImplementedError()
    
    def device_mem_alloc(self, size):
        return TorchDeviceBuffer(size)

def align_size(self, size, alignment=256):
    return ((size + alignment - 1) // alignment) * alignment