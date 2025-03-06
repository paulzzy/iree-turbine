# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    Optional,
    Union,
    List,
)

from pathlib import Path

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

from iree.runtime import (
    VmModule,
)

from iree.compiler.extras.fx_importer import FxImporter

import torch
from ..passes import turbine_cpu_pass_pipeline


def backend_generator(
    *,
    target_backends: Union[List[str], str],
    flags: List[str] = [],
    driver: str,
    save_mlir: Optional[Union[str, Path]] = None,
):
    """
    Users can call this directly with:
        ```
        from iree.turbine.dynamo.backends.basic import backend_generator
        def my_func(...):
            ...
        my_backend = backend_generator(
            target_backends=...,
            flags=...,
            driver=...,
            )
        compiled_func = torch.compile(my_func, my_backend)
        y = compiled_func(*args)
        ```
    - target_backends mirror the options for `--iree-hal-target-backends` in `iree-compile`
    - flags are for passing a list of `iree-compile` flags
    - driver is for running the compiled artifact. E.g., `driver=local-task` for cpu or `driver=hip` for amd gpu
    - save_mlir is for dumping the imported IR to a specified file.
    """

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

        # Save the imported mlir file if requested
        if save_mlir:
            p = Path(save_mlir)
            p.write_text(str(module))
        else:
            print(module, file=sys.stderr)

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
