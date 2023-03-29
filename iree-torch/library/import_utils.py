import io
from typing import Any, Tuple
import torch
import torch_mlir

from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


def _strip_overloads(gm):
  """Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
  for node in gm.graph.nodes:
    if isinstance(node.target, torch._ops.OpOverload):
      node.target = node.target.overloadpacket
  gm.recompile()


def import_torch_module(module: torch.nn.Module, inputs: Tuple[Any, ...],
                        output_dialect: torch_mlir.OutputType):
  mlir_module = torch_mlir.compile(module, inputs, output_type=output_dialect)
  bytecode_stream = io.BytesIO()
  mlir_module.operation.write_bytecode(bytecode_stream)
  return bytecode_stream.getvalue()


def import_torch_module_with_fx(module: torch.nn.Module, inputs: Tuple[Any,
                                                                       ...],
                                output_dialect: torch_mlir.OutputType):
  fx_g = make_fx(
      module,
      decomposition_table=get_decompositions([
          torch.ops.aten.embedding_dense_backward,
          torch.ops.aten.native_layer_norm_backward,
          torch.ops.aten.slice_backward,
          torch.ops.aten.select_backward,
          torch.ops.aten.norm.ScalarOpt_dim,
          torch.ops.aten.native_group_norm,
          torch.ops.aten.upsample_bilinear2d.vec,
          torch.ops.aten.split.Tensor,
          torch.ops.aten.split_with_sizes,
          torch.ops.aten.native_layer_norm,
      ]),
  )(*inputs)

  fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
  fx_g.recompile()

  _strip_overloads(fx_g)
  ts_graph = torch.jit.script(fx_g)
  return import_torch_module(ts_graph, inputs, output_dialect)
