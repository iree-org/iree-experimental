LLVM_SRC=$(HOME)/github/llvm-project
LLVM_BUILD=$(HOME)/github/llvm-project/build
IREE_SRC=$(HOME)/github/iree
IREE_BUILD=$(HOME)/github/iree/build
IREE_SAMPLES_DIR=$(HOME)/github/iree-samples/transform_dialect/examples

MLIR_OPT=$(LLVM_BUILD)/bin/mlir-opt
IREE_OPT=$(IREE_BUILD)/tools/iree-opt
IREE_COMPILE=$(IREE_BUILD)/tools/iree-compile
IREE_RUN_MODULE=$(IREE_BUILD)/tools/iree-run-module

MATMUL_STUB=$(IREE_SAMPLES_DIR)/matmul.mlir


.PHONY: all


all: fill_matmul


define extract_size_from_matmul
  $(subst _, ,$(shell echo $1 | sed -e "s/.*_matmul_//g" | sed -e "s/\.mlir//g" | sed -e "s/\.td//g"))
endef


tmp_fill_matmul_%.mlir: $(MATMUL_STUB)
	$(eval M:=$(word 1, $(call extract_size_from_matmul,$@)))
	$(eval N:=$(word 2, $(call extract_size_from_matmul,$@)))
	$(eval K:=$(word 3, $(call extract_size_from_matmul,$@)))
	@cat $(MATMUL_STUB) | sed "s/\$${M}/$M/g" | sed "s/\$${N}/$N/g" | sed "s/\$${K}/$K/g" > $@;


fill_matmul_%.mlir: tmp_fill_matmul_%.mlir $(MLIR_OPT)
	@cat $< | sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
    $(MLIR_OPT) -symbol-dce > $@


iree_cuda_fill_matmul_%.vmfb: fill_matmul_%.mlir $(IREE_COMPILE)
	@$(IREE_COMPILE) $< --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
	  --iree-hal-benchmark-dispatch-repeat-count=3 \
		-o $@


iree_cuda_fill_matmul_%.td.vmfb: fill_matmul_%.mlir $(IREE_COMPILE)
	@$(IREE_COMPILE) $< --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
	  --iree-hal-benchmark-dispatch-repeat-count=3 \
		--iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy \
		-o $@


run_%: %.vmfb
	@echo Processing $<
	$(eval M:=$(word 1, $(call extract_size_from_matmul,$@)))
	$(eval N:=$(word 2, $(call extract_size_from_matmul,$@)))
	$(eval K:=$(word 3, $(call extract_size_from_matmul,$@)))
	@printf '%-50s' "$@" > logs/$@.log
	@printf '%s' " -> " >> logs/$@.log
	@{ \
		scp $< $(USER)@$(A100_MACHINE_IP):~/ > /dev/null 2>&1; \
		ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=fill_matmul_static --device=cuda  --module=$< --input=$(M)x$(K)xf32=1 --input=$(K)x$(N)xf32=1 --input=$(M)x$(N)xf32=1 2>&1" | \
		grep fill_matmul_static | grep -v EXEC | \
		awk '{print $$6}'; \
	} >> logs/$@.log


fill_matmul_%: run_iree_cuda_fill_matmul_% run_iree_cuda_fill_matmul_%.td
	@echo Done with $@

logs:
	mkdir -p logs

some_tests: fill_matmul_3452_1000_1000 fill_matmul_2052_2052_2052 fill_matmul_1000_1000_3052 logs
	ls logs/*.log | sort -n | xargs cat


clean:
	rm -f *.vmfb logs/*
	rmdir logs
