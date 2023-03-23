add_mlir_conversion_library(MLIRCUDNNToLLVM
  CUDNNToLLVM.cpp

  DEPENDS
  CUDNNConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRCUDNNDialect
  MLIRLLVMDialect
  )
