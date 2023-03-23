add_mlir_dialect_library(MLIRCUDNNDialect
        CUDNNDialect.cpp
        CUDNNOps.cpp
        CUDNNTypes.cpp

        DEPENDS
        MLIRCUDNNOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
