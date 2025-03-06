# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from torch._dynamo.backends.common import aot_autograd
from basic import backend_generator

basic_cpu_options = {
    "target_backends": "llvm-cpu",
    "flags": ["--iree-llvmcpu-target-cpu=host"],
    "driver": "local-task",
}

backend = aot_autograd(fw_compiler=backend_generator(**basic_cpu_options))
