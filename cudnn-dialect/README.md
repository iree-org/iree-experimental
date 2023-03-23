# cuDNN dialect

**🚨 This is an early-stage project. All details are subject to arbitrary changes/open to discussion. 🚨**

The cuDNN dialect represents (primarily) operations in a graph towards creating
an operation graph description. The dialect matches cuDNN specification, and
serves as target that is simple/directly useful for export rather than
higher-level convenience (for which there are other dialects, including TCP).
The operation grouping follows the example of the frontend API, but lowering
via LLVM dialect targets the backend API.

This was pinned to LLVM commit 7ccbb4d during development and follows style of
https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone , it is
expected that cuDNN is installed for converting to LLVM dialect.
