import data_types
import input_data_definitions
import jax_output_data_definitions
import unique_ids

from typing import List

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684283564"

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_LARGE_FP32_JAX_384XI32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1,
    name="BERT_LARGE_JAX_384XI32_BATCH1",
    tags=["fp32", "transformer-encoder", "bert", "batch-1"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=1,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH1,
    outputs=jax_output_data_definitions.BERT_LARGE_FP32_JAX_384X1024XF32_BATCH1,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH16 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH16,
    name="BERT_LARGE_JAX_384XI32_BATCH16",
    tags=["fp32", "transformer-encoder", "bert", "batch-16"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=16,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH16,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH16,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_16/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_16/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH24 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH24,
    name="BERT_LARGE_JAX_384XI32_BATCH24",
    tags=["fp32", "transformer-encoder", "bert", "batch-24"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=24,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH24,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH24,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_24/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_24/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH32 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH32,
    name="BERT_LARGE_JAX_384XI32_BATCH32",
    tags=["fp32", "transformer-encoder", "bert", "batch-32"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=32,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH32,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH32,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_32/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_32/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH48 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH48,
    name="BERT_LARGE_JAX_384XI32_BATCH48",
    tags=["fp32", "transformer-encoder", "bert", "batch-48"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=48,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH48,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH48,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_48/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_48/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH64,
    name="BERT_LARGE_JAX_384XI32_BATCH64",
    tags=["fp32", "transformer-encoder", "bert", "batch-64"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=64,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH64,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH64,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_64/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_64/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH512 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH512,
    name="BERT_LARGE_JAX_384XI32_BATCH512",
    tags=["fp32", "transformer-encoder", "bert", "batch-512"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=512,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH512,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH512,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_512/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_512/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH1024 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1024,
    name="BERT_LARGE_JAX_384XI32_BATCH1024",
    tags=["fp32", "transformer-encoder", "bert", "batch-1024"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=1024,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH1024,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH1024,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1024/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1024/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH1280 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1280,
    name="BERT_LARGE_JAX_384XI32_BATCH1280",
    tags=["fp32", "transformer-encoder", "bert", "batch-1280"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    input_batch_size=1280,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH1280,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCH1280,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1280/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1280/hlo/module_0081.jit_model_jitted.before_optimizations.txt",
        ),
    ],
)

# Dictionaries.
JAX_MODELS_DICT = {
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1:
        BERT_LARGE_FP32_JAX_384XI32_BATCH1,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH16:
        BERT_LARGE_FP32_JAX_384XI32_BATCH16,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH24:
        BERT_LARGE_FP32_JAX_384XI32_BATCH24,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH32:
        BERT_LARGE_FP32_JAX_384XI32_BATCH32,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH48:
        BERT_LARGE_FP32_JAX_384XI32_BATCH48,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH64:
        BERT_LARGE_FP32_JAX_384XI32_BATCH64,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH512:
        BERT_LARGE_FP32_JAX_384XI32_BATCH512,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1024:
        BERT_LARGE_FP32_JAX_384XI32_BATCH1024,
    unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1280:
        BERT_LARGE_FP32_JAX_384XI32_BATCH1280,
}
