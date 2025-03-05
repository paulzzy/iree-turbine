# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import sys

from ...runtime.device import (
    DeviceState,
)

from ..executor import (
    SpecializedExecutable,
)

from iree.compiler.api import (
    Invocation,
    Session,
    Source,
    Output,
)

from iree.compiler.ir import (
    Context,
)
from iree.compiler.passmanager import (
    PassManager,
)

from iree.compiler.tools import (
    CompilerOptions,
)

from iree.runtime import (
    VmModule,
)

from iree.compiler.extras.fx_importer import FxImporter

import torch
from torch._dynamo.backends.common import aot_autograd
from ..passes import turbine_cpu_pass_pipeline


def _base_backend(gm: torch.fx.GraphModule, example_inputs, **options):
    # Set up the session, context and invocation.
    # Note that we do this on one in-memory module in a few phases:
    #  1. Build it from the FX graph.
    #  2. Run torch MLIR passes to lower it to a suitable form for
    #     input.
    #  3. Run IREE's main compiler.
    #  4. Output to an mmap buffer.
    target_backends = options.get("target_backends", None)
    flags = options.get("compiler_flags", [])
    device = options.get("device", "hip")

    if isinstance(target_backends, str):
        flags.append(f"--iree-hal-target-backends={target_backends}")
    elif isinstance(target_backends, list[str]):
        flags.extend(
            [f"--iree-hal-target-backends={backend}" for backend in target_backends]
        )
    session = Session()
    if len(flags) > 0:
        session.set_flags(flags)
    context = session.context
    importer = FxImporter(context=context)
    module = importer.module
    inv = session.invocation()
    # TODO: Should capture diagnostics.
    inv.enable_console_diagnostics()
    inv.import_module(module.operation)

    # Apply decompositions.
    gm = turbine_cpu_pass_pipeline(gm, example_inputs)

    # Import phase.
    importer.import_graph_module(gm)
    print(module, file=sys.stderr)

    # IREE compilation phase.
    inv.execute()

    # Output phase.
    output = Output.open_membuffer()
    inv.output_vm_bytecode(output)

    # Set up for runtime.
    device_state = _get_device_state(device)
    vmfb_module = VmModule.wrap_buffer(
        device_state.instance,
        output.map_memory(),
        destroy_callback=output.close,
    )
    output.close()

    return SpecializedExecutable(vmfb_module, device_state)


backend = aot_autograd(fw_compiler=_base_backend)


# IREE runtime globals.
@functools.lru_cache(maxsize=None)
def _get_device_state(driver) -> DeviceState:
    # how can I automatically detect
    return DeviceState(driver=driver)
