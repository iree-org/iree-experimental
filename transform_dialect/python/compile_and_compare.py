import argparse

# Torch used to make value comparison times bearable.
import torch

import iree.runtime as ireert
from iree.runtime import get_driver, get_device
import iree.compiler as ireec


def run_vmfb(vmfb, function_name, runtime_device, inputs):
    config = ireert.Config(driver_name=runtime_device)
    vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, vmfb)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    forward = ctx.modules.module[function_name]
    return forward(*inputs).to_host()


def compare_mlir_module(
    module,
    function_name,
    device,
    inputs,
    target_info_flag,
    argset1=[],
    argset2=[],
    runtime_device=None,
):
    if runtime_device is None:
        runtime_device = device

    # These flags are almost always wanted.
    base_args = [
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        target_info_flag,
    ]

    flatbuffer = ireec.compile_str(
        module,
        target_backends=[device],
        extra_args=base_args + argset1,
    )
    iree_output1 = run_vmfb(flatbuffer, function_name, runtime_device, inputs)

    flatbuffer = ireec.compile_str(
        module,
        target_backends=[device],
        extra_args=base_args + argset2,
    )
    iree_output2 = run_vmfb(flatbuffer, function_name, runtime_device, inputs)

    result1 = torch.from_numpy(iree_output1)
    result2 = torch.from_numpy(iree_output2)

    # TODO make tolerances configurable.
    torch.testing.assert_close(result1, result2, rtol=1e-03, atol=4.1e-02)


def compare_vmfbs(vmfb1, vmfb2, function_name, runtime_device, inputs):
    iree_output1 = run_vmfb(vmfb1, function_name, runtime_device, inputs)
    iree_output2 = run_vmfb(vmfb2, function_name, runtime_device, inputs)

    result1 = torch.from_numpy(iree_output1)
    result2 = torch.from_numpy(iree_output2)

    # TODO make tolerances configurable.
    torch.testing.assert_close(result1, result2, rtol=1e-03, atol=4.1e-02)


if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="IREE vmfb comparator.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target for compilation.",
    )
    parser.add_argument(
        "--runtime_device",
        type=str,
        default=None,
        help="""Device to feed to the runtime. If not specified
                    will just use the device for compilation.""",
    )
    parser.add_argument(
        "--module",
        type=str,
        default=None,
        help="Path to the input mlir.",
    )
    parser.add_argument(
        "--function",
        type=str,
        default="forward",
        help="Name of the function to run from the input mlir.",
    )
    parser.add_argument(
        "--target_info_flag",
        type=str,
        default="--iree-hal-cuda-llvm-target-arch=sm_80",
        help="Flag the specify the target architecture to compile for.",
    )
    parser.add_argument(
        "--input_types",
        type=str,
        help='List of input types, e.g. "1x2x3xf32 2x3x3xf16"',
    )
    parser.add_argument(
        "--argset1",
        type=str,
        default="",
        help='First set of compiler arguments to use.'
    )
    parser.add_argument(
        "--argset2",
        type=str,
        default="",
        help='Second set of compiler arguments to use.'
    )
    parser.add_argument(
        "--vmfb1_path",
        type=str,
        default=None,
        help="""Path to a vmfb for comparison. Requires vmfb2_path
                    as well. Mutually exclusive with --module""",
    )
    parser.add_argument(
        "--vmfb2_path",
        type=str,
        default=None,
        help="""Path to a vmfb for comparison. Requires vmfb1_path
                    as well. Mutually exclusive with --module""",
    )

    args, unknown = parser.parse_known_args()

    runtime_device = args.runtime_device
    if runtime_device is None:
        runtime_device = args.device

    input_types_str = args.input_types
    inputs = []
    for input_type in input_types_str.split():
        shape_and_dtype = input_type.split("x")
        shape = shape_and_dtype[:-1]
        dtype = shape_and_dtype[-1]
        # TODO: Support more dtypes.
        if dtype == "f32":
            torch_dtype = torch.float32
        elif dtype == "f16":
            torch_dtype = torch.float16
        else:
            raise Exception("Unsupported dtype")

        # TODO: Be smarter about initial numerical values.
        inputs.append(torch.randn([int(s) for s in shape], dtype=torch_dtype))

    if args.module is not None:
        if args.vmfb1_path is not None or args.vmfb2_path is not None:
            raise Exception(
                "Option clash, cannot specify an mlir module and a vmfb"
            )

        with open(args.module, "rb") as f:
            module = f.read()
        compare_mlir_module(
            module,
            args.function,
            args.device,
            inputs,
            args.target_info_flag,
            args.argset1.split(),
            args.argset2.split(),
            runtime_device,
        )

    elif args.vmfb1_path is not None:
        if args.vmfb2_path is None:
            raise Exception("Two vmfbs must be specified for comparison")

        with open(args.vmfb1_path, "rb") as f:
            flatbuffer1 = f.read()
        with open(args.vmfb2_path, "rb") as f:
            flatbuffer2 = f.read()

        compare_vmfbs(
            flatbuffer1,
            flatbuffer2,
            args.function,
            runtime_device,
            inputs,
        )

    else:
      print("See usage with --help")
      quit()
    print("Success!")
