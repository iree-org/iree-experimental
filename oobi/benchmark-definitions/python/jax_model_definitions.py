import data_types
import input_data_definitions
import jax_output_data_definitions
import unique_ids

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752"

# Meta models.
RESNET50_FP32_JAX = data_types.MetaModel(
    id=unique_ids.MODEL_RESNET50_FP32_JAX,
    name="RESNET50_FP32_JAX",
    tags=["fp32", "cnn", "resnet"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel",
    data_type=data_types.DataType.FP32,
)

BERT_LARGE_FP32_JAX = data_types.MetaModel(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX,
    name="BERT_LARGE_FP32_JAX",
    tags=["fp32", "transformer-encoder", "bert"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    data_type=data_types.DataType.FP32,
)

T5_LARGE_FP32_JAX = data_types.MetaModel(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX,
    name="T5_LARGE_FP32_JAX",
    tags=["fp32", "transformer-encoder", "transformer-decoder", "t5"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model",
    data_type=data_types.DataType.FP32,
)


# Resnet50 models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
RESNET50_FP32_JAX_3X224X224XF32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH1,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH1",
    tags=["batch-1"],
    meta_model=RESNET50_FP32_JAX,
    input_batch_size=1,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH1,
    outputs=jax_output_data_definitions.RESNET50_FP32_JAX_2048X7X7XF32_BATCH1,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_1/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/RESNET50/batch_1/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

RESNET50_FP32_JAX_3X224X224XF32_BATCH8 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH8,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH8",
    tags=["batch-8"],
    meta_model=RESNET50_FP32_JAX,
    input_batch_size=8,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH8,
    outputs=jax_output_data_definitions.RESNET50_FP32_JAX_2048X7X7XF32_BATCH8,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_8/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/RESNET50/batch_8/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

RESNET50_FP32_JAX_3X224X224XF32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH64,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH64",
    tags=["batch-64"],
    meta_model=RESNET50_FP32_JAX,
    input_batch_size=64,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH64,
    outputs=jax_output_data_definitions.RESNET50_FP32_JAX_2048X7X7XF32_BATCH64,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_64/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/RESNET50/batch_64/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

RESNET50_FP32_JAX_3X224X224XF32_BATCH128 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH128,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH128",
    tags=["batch-128"],
    meta_model=RESNET50_FP32_JAX,
    input_batch_size=128,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH128,
    outputs=jax_output_data_definitions.RESNET50_FP32_JAX_2048X7X7XF32_BATCH128,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_128/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/RESNET50/batch_128/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

RESNET50_FP32_JAX_3X224X224XF32_BATCH256 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH256,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH256",
    tags=["batch-256"],
    meta_model=RESNET50_FP32_JAX,
    input_batch_size=256,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH256,
    outputs=jax_output_data_definitions.RESNET50_FP32_JAX_2048X7X7XF32_BATCH256,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_256/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/RESNET50/batch_256/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

RESNET50_FP32_JAX_3X224X224XF32_BATCH2048 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH2048,
    name="RESNET50_FP32_JAX_3X224X224XF32_BATCH2048",
    tags=["batch-2048"],
    meta_model=RESNET50_FP32_JAX,
    input_batch_size=2048,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH2048,
    outputs=jax_output_data_definitions.
    RESNET50_FP32_JAX_2048X7X7XF32_BATCH2048,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_2048/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/RESNET50/batch_2048/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)


# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_LARGE_FP32_JAX_384XI32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1,
    name="BERT_LARGE_JAX_384XI32_BATCH1",
    tags=["batch-1"],
    meta_model=BERT_LARGE_FP32_JAX,
    input_batch_size=1,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH1,
    outputs=jax_output_data_definitions.BERT_LARGE_FP32_JAX_384X1024XF32_BATCH1,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH16 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH16,
    name="BERT_LARGE_JAX_384XI32_BATCH16",
    tags=["batch-16"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_16/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH24 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH24,
    name="BERT_LARGE_JAX_384XI32_BATCH24",
    tags=["batch-24"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_24/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH32 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH32,
    name="BERT_LARGE_JAX_384XI32_BATCH32",
    tags=["batch-32"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_32/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH48 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH48,
    name="BERT_LARGE_JAX_384XI32_BATCH48",
    tags=["batch-48"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_48/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH64,
    name="BERT_LARGE_JAX_384XI32_BATCH64",
    tags=["batch-64"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_64/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH512 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH512,
    name="BERT_LARGE_JAX_384XI32_BATCH512",
    tags=["batch-512"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_512/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH1024 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1024,
    name="BERT_LARGE_JAX_384XI32_BATCH1024",
    tags=["batch-1024"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1024/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

BERT_LARGE_FP32_JAX_384XI32_BATCH1280 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32_BATCH1280,
    name="BERT_LARGE_JAX_384XI32_BATCH1280",
    tags=["batch-1280"],
    meta_model=BERT_LARGE_FP32_JAX,
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
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1280/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

# T5-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model
# Bert-Large batch sizes used for T5-Large models.
T5_LARGE_FP32_JAX_512XI32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH1,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH1",
    tags=["batch-1"],
    meta_model=T5_LARGE_FP32_JAX,
    input_batch_size=1,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH1,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCH1,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_1/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/T5_LARGE/batch_1/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

T5_LARGE_FP32_JAX_512XI32_BATCH16 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH16,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH16",
    tags=["batch-16"],
    meta_model=T5_LARGE_FP32_JAX,
    input_batch_size=16,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH16,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCH16,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_16/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/T5_LARGE/batch_16/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

T5_LARGE_FP32_JAX_512XI32_BATCH24 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH24,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH24",
    tags=["batch-24"],
    meta_model=T5_LARGE_FP32_JAX,
    input_batch_size=24,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH24,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCH24,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_24/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/T5_LARGE/batch_24/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

T5_LARGE_FP32_JAX_512XI32_BATCH32 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH32,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH32",
    tags=["batch-32"],
    meta_model=T5_LARGE_FP32_JAX,
    input_batch_size=32,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH32,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCH32,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_32/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/T5_LARGE/batch_32/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

T5_LARGE_FP32_JAX_512XI32_BATCH48 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH48,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH48",
    tags=["batch-48"],
    meta_model=T5_LARGE_FP32_JAX,
    input_batch_size=48,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH48,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCH48,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_48/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/T5_LARGE/batch_48/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

T5_LARGE_FP32_JAX_512XI32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH64,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH64",
    tags=["batch-64"],
    meta_model=T5_LARGE_FP32_JAX,
    input_batch_size=64,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH64,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCH64,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_64/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/T5_LARGE/batch_64/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

T5_LARGE_FP32_JAX_512XI32_BATCH512 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH512,
    name="T5_LARGE_FP32_JAX_512XI32_BATCH512",
    tags=["batch-512"],
    meta_model=T5_LARGE_FP32_JAX,
    input_batch_size=512,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH512,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCH512,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_512/stablehlo.mlirbc",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=
            f"{PARENT_GCS_DIR}/T5_LARGE/batch_512/hlo/jit_forward.before_optimizations.txt",
        ),
    ],
)

# Dictionaries.
JAX_MODELS_DICT = {
    unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH1:
        RESNET50_FP32_JAX_3X224X224XF32_BATCH1,
    unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH8:
        RESNET50_FP32_JAX_3X224X224XF32_BATCH8,
    unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH64:
        RESNET50_FP32_JAX_3X224X224XF32_BATCH64,
    unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH128:
        RESNET50_FP32_JAX_3X224X224XF32_BATCH128,
    unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH256:
        RESNET50_FP32_JAX_3X224X224XF32_BATCH256,
    unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32_BATCH2048:
        RESNET50_FP32_JAX_3X224X224XF32_BATCH2048,
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
    unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH1:
        T5_LARGE_FP32_JAX_512XI32_BATCH1,
    unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH16:
        T5_LARGE_FP32_JAX_512XI32_BATCH16,
    unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH24:
        T5_LARGE_FP32_JAX_512XI32_BATCH24,
    unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH32:
        T5_LARGE_FP32_JAX_512XI32_BATCH32,
    unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH48:
        T5_LARGE_FP32_JAX_512XI32_BATCH48,
    unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH64:
        T5_LARGE_FP32_JAX_512XI32_BATCH64,
    unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32_BATCH512:
        T5_LARGE_FP32_JAX_512XI32_BATCH512,
}
