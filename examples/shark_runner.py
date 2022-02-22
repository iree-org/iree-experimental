import torch
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir_e2e_test.torchscript.annotations import (
    extract_serializable_annotations,
    apply_serializable_annotations,
)

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.torchscript.configs import LinalgOnTensorsBackendTestConfig
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from torch_mlir_e2e_test.torchscript.framework import (
    SerializableTest,
    generate_golden_trace,
)
from iree import runtime as ireert
import iree.compiler as ireec
import numpy as np
import os
import sys
import pickle
import io


def getTracedRecursiveScriptModule(module, input, dynamic):
    input_shape = list(input.shape)
    if dynamic:
        input_shape = [-1 for i in range(len(input.shape))]

    traced_module = torch.jit.trace_module(module, {"forward": input})
    script_module = traced_module._actual_script_module
    export(script_module.forward)
    annotate_args_decorator = annotate_args(
        [
            None,
            (input_shape, input.dtype, True),
        ]
    )
    annotate_args_decorator(script_module.forward)
    return script_module


def get_serialized_test(model):
    module = torch.jit.script(model)
    torchscript_module_bytes = module.save_to_buffer(
        {"annotations.pkl": pickle.dumps(extract_serializable_annotations(module))}
    )
    serializable_test = SerializableTest(
        unique_name="", program=torchscript_module_bytes, trace=None
    )
    _extra_files = {"annotations.pkl": ""}
    module = torch.jit.load(
        io.BytesIO(serializable_test.program), _extra_files=_extra_files
    )
    # Load the pickled annotations.
    annotations = pickle.loads(_extra_files["annotations.pkl"])
    apply_serializable_annotations(module, annotations)
    return module


# shark_inference takes torch.nn.module, input, and device, extracts torchscript,
# runs the torch-mlir linalg on tensors backend and finally runs via iree stack.
def shark_inference(module, input, device="cpu", dynamic=False, trace_module=False):

    if trace_module:
        module = getTracedRecursiveScriptModule(module, input, dynamic)
        module = get_serialized_test(module)
    else:
        module = torch.jit.script(module)

    input_shape = list(input.shape)
    if dynamic:
        input_shape = [-1 for i in range(len(input.shape))]

    mb = ModuleBuilder()
    class_annotator = ClassAnnotator()
    iree_device = {"cpu": "dylib", "gpu": "cuda", "vulkan": "vulkan"}
    class_annotator.exportNone(module._c._type())
    class_annotator.exportPath(module._c._type(), ["forward"])
    class_annotator.annotateArgs(
        module._c._type(),
        ["forward"],
        [
            None,
            (input_shape, input.dtype, True),
        ],
    )
    mb.import_module(module._c, class_annotator)

    with mb.module.context:
        pm = PassManager.parse(
            "torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline"
        )
        pm.run(mb.module)

    flatbuffer_blob = ireec.compile_str(
        str(mb.module), target_backends=[iree_device[device]]
    )
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    tracer = ireert.Tracer(os.getcwd())
    config = ireert.Config(iree_device[device], tracer)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module["forward"]
    result = ModuleCompiled(input.numpy())
    return np.asarray(result, dtype=np.float32)
