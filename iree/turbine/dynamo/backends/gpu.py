# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    Union,
    List,
)

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

from iree.runtime import (
    VmModule,
)

from iree.compiler.extras.fx_importer import FxImporter

import torch
from ..passes import turbine_cpu_pass_pipeline


def basic_turbine(
    *,
    target_backends: Union[List[str], str],
    flags: List[str] = [],
    driver: str,
):
    def _backend(gm: torch.export.ExportedProgram, example_inputs):
        # Set up the session, context and invocation.
        # Note that we do this on one in-memory module in a few phases:
        #  1. Build it from the FX graph.
        #  2. Run IREE's main compiler.
        #  3. Output to an mmap buffer.

        if isinstance(target_backends, str):
            flags.append(f"--iree-hal-target-backends={target_backends}")
        elif isinstance(target_backends, list):
            flags.extend(
                [f"--iree-hal-target-backends={backend}" for backend in target_backends]
            )
        session = Session()
        if len(flags) > 0:
            session.set_flags(*flags)
        context = session.context
        importer = FxImporter(context=context)
        module = importer.module
        inv = session.invocation()
        # TODO: Should capture diagnostics.
        inv.enable_console_diagnostics()
        inv.import_module(module.operation)

        # Apply decompositions (maybe rename this function).
        gm = turbine_cpu_pass_pipeline(gm, example_inputs)

        # Import phase.
        importer.import_graph_module(gm)

        # IREE compilation phase.
        inv.execute()

        # Output phase.
        output = Output.open_membuffer()
        inv.output_vm_bytecode(output)

        # Set up for runtime.
        device_state = DeviceState.from_uri(driver)
        vmfb_module = VmModule.copy_buffer(
            device_state.instance,
            output.map_memory(),
        )
        output.close()

        return SpecializedExecutable(vmfb_module, device_state)

    return _backend
