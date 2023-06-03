# module-wide variables
import os
import sys
this = sys.modules[__name__]

from cutlass.backend.arguments import *
from cutlass.backend.c_types import *
from cutlass.backend.compiler import ArtifactManager, CompilationOptions
from cutlass.backend.conv2d_operation import *
from cutlass.backend.epilogue import *
from cutlass.backend.frontend import *
from cutlass.backend.gemm_operation import *
from cutlass.backend.library import *
from cutlass.backend.memory_manager import PoolMemoryManager, TorchMemoryManager
from cutlass.backend.operation import *
from cutlass.backend.parser import *
from cutlass.backend.reduction_operation import *
from cutlass.backend.tensor_ref import *
from cutlass.backend.type_hint import *
from cutlass.backend.utils import *
from cutlass.backend.utils.device import device_cc
from cutlass.backend.utils.software import (
    CheckPackages,
    SubstituteTemplate,
    device_sm_count,
)

compiler = ArtifactManager()

def get_memory_pool(manager="rmm", init_pool_size=0, max_pool_size=2**34):
    if manager == "rmm":
        this.memory_pool = PoolMemoryManager(
            init_pool_size=init_pool_size,
            max_pool_size=max_pool_size
        )
    elif manager == "torch":
        this.memory_pool = TorchMemoryManager()
    return this.memory_pool
