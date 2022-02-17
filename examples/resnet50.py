from PIL import Image
import requests
import torch
import torchvision.models as models
from torchvision import transforms

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.torchscript.configs import LinalgOnTensorsBackendTestConfig
from iree import runtime as ireert
import iree.compiler as ireec
import numpy as np
import os
import sys


################################## Preprocessing inputs and model ############
def load_and_preprocess_image(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    img = Image.open(requests.get(url, headers=headers, stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3


class Resnet50Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = Resnet50Module()

    def forward(self, x):
        return self.s.forward(x)


image_url = (
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
)
print("load image from " + image_url, file=sys.stderr)
img = load_and_preprocess_image(image_url)
labels = load_labels()

##############################################################################

# inference_iree takes torch.nn.module, input, and device, extracts torchscript,
# runs the torch-mlir linalg on tensors backend and finally runs via iree stack.
def inference_iree(module, input, device):
    mb = ModuleBuilder()
    class_annotator = ClassAnnotator()
    iree_device = {"cpu": "dylib", "gpu": "cuda", "vulkan": "vulkan"}
    recursivescriptmodule = torch.jit.script(module)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(),
        ["forward"],
        [
            None,
            (list(input.shape), input.dtype, True),
        ],
    )
    mb.import_module(recursivescriptmodule._c, class_annotator)

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
results = inference_iree(TestModule(), img, "cpu")

print("The top 3 results obtained via torch is:")
print(top3_possibilities(TestModule()(img)))
print()
print("The top 3 results obtained via torch-mlir via iree-backend is:")
print(top3_possibilities(torch.from_numpy(results)))
