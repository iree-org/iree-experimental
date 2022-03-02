# Shark Runner

The Shark Runner provides inference and training APIs to run deep learning models on Shark Runtime.

# How to configure.

### Build [torch-mlir](https://github.com/llvm/torch-mlir) and [iree](https://github.com/google/iree).

### Setup Python Environment
```shell
export PYTHONPATH={torch-mlir-build-dir}/tools/torch-mlir/python_packages/torch_mlir
```
export [iree python bindings](https://google.github.io/iree/building-from-source/python-bindings-and-importers/#using-the-python-bindings)


### Shark Inference API

```
from shark_runner import shark_inference

result = shark_inference(
        module = torch.nn.module class.
        input  = input to model (must be a torch-tensor)
        device = `cpu`, `gpu` or `vulkan` is supported.
        dynamic (boolean) = Pass the input shapes as static or dynamic.
        jit_trace (boolean) = Jit trace the module with the given input, useful in the case where jit.script doesn't work. )
```
