from PIL import Image
import requests
import torch
import torchvision.models as models
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir_e2e_test.torchscript.annotations import extract_serializable_annotations, apply_serializable_annotations

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.torchscript.configs import LinalgOnTensorsBackendTestConfig
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from iree import runtime as ireert
import iree.compiler as ireec
import numpy as np
import os
import sys

torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")


def _prepare_sentence_tokens(sentence: str):
    return torch.tensor([tokenizer.encode(sentence)])

class MiniLMSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",  # The pretrained model.
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True)

    @export
    @annotate_args([
        None,
        ([1, 7], torch.int64, True),
    ])
    def forward(self, tokens):
        return self.model.forward(tokens)[0]

test_input = _prepare_sentence_tokens("this project is very interesting")

def getTracedRecursiveScriptModule():
    traced_module = torch.jit.trace_module(MiniLMSequenceClassification(),
            {'forward': test_input})
    script_module = traced_module._actual_script_module
    export(script_module.forward)
    annotate_args_decorator = annotate_args([
        None,
        ([1, 7], torch.int64, True),
    ])
    annotate_args_decorator(script_module.forward)
    return script_module


# inference_iree takes torch.nn.module, input, and device, extracts torchscript,
# runs the torch-mlir linalg on tensors backend and finally runs via iree stack.
def inference_iree(module, input, device):
    # traced_module = torch.jit.script(MiniLMSequenceClassification())
    mb = ModuleBuilder()
    class_annotator = ClassAnnotator()
    iree_device = {"cpu": "dylib", "gpu": "cuda", "vulkan": "vulkan"}
    recursivescriptmodule = getTracedRecursiveScriptModule()
    annotations = extract_serializable_annotations(recursivescriptmodule)
    apply_serializable_annotations(recursivescriptmodule, annotations)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    # class_annotator.annotateArgs(
        # recursivescriptmodule._c._type(),
        # ["forward"],
        # [
            # None,
            # ([1, 7], torch.int64, True),
        # ],
    # )
    mb.import_module(recursivescriptmodule._c, class_annotator)
    mb.module.dump()

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
    result = ModuleCompiled(img.numpy())
    return np.asarray(result, dtype=np.float32)


## The device should be "cpu", "gpu" or "vulkan".
results = inference_iree(MiniLMSequenceClassification(), test_input, "cpu")

print("The top 3 results obtained via torch is:")
print(top3_possibilities(MiniLMSequenceClassification()(test_input)))
print()
print("The top 3 results obtained via torch-mlir via iree-backend is:")
print(top3_possibilities(torch.from_numpy(results)))
