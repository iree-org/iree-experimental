get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIRCUDNNDialect
        MLIROptLib
        )
add_llvm_executable(cudnn-opt cudnn-opt.cpp)

llvm_update_compile_flags(cudnn-opt)
target_link_libraries(cudnn-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(cudnn-opt)
