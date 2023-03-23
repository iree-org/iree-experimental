add_mlir_dialect(CUDNNOps cudnn)
add_mlir_doc(CUDNN CUDNN Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS CUDNNTypes.td)
mlir_tablegen(CUDNNTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CUDNNTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRCUDNNTypesIncGen)
add_dependencies(mlir-headers MLIRCUDNNTypesIncGen)
