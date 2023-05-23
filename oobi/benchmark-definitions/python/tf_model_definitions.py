import data_types
import unique_ids

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794"

# Meta models.
RESNET50_FP32_TF = data_types.MetaModel(
    id=unique_ids.MODEL_RESNET50_FP32_TF,
    name="RESNET50_FP32_TF",
    tags=["fp32", "cnn", "resnet"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
    data_type=data_types.DataType.FP32,
)

BERT_LARGE_FP32_TF = data_types.MetaModel(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF,
    name="BERT_LARGE_FP32_TF",
    tags=["fp32", "transformer-encoder", "bert"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    data_type=data_types.DataType.FP32,
)

T5_LARGE_FP32_TF = data_types.MetaModel(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF,
    name="T5_LARGE_FP32_TF",
    tags=["fp32", "transformer-encoder", "transformer-decoder", "t5"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    data_type=data_types.DataType.FP32,
)

# Resnet50 models.
# Model implementation from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
RESNET50_FP32_TF_224X224X3XF32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH1,
    name="RESNET50_FP32_TF_224X224X3XF32_BATCH1",
    tags=["batch-1"],
    meta_model=RESNET50_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_1/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_1/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_1/hlo.mlirbc",
        ),
    ],
)

RESNET50_FP32_TF_224X224X3XF32_BATCH8 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH8,
    name="RESNET50_FP32_TF_224X224X3XF32_BATCH8",
    tags=["batch-8"],
    meta_model=RESNET50_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_8/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_8/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_8/hlo.mlirbc",
        ),
    ],
)

RESNET50_FP32_TF_224X224X3XF32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH64,
    name="RESNET50_FP32_TF_224X224X3XF32_BATCH64",
    tags=["batch-64"],
    meta_model=RESNET50_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_64/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_64/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_64/hlo.mlirbc",
        ),
    ],
)

RESNET50_FP32_TF_224X224X3XF32_BATCH128 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH128,
    name="RESNET50_FP32_TF_224X224X3XF32_BATCH128",
    tags=["batch-128"],
    meta_model=RESNET50_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_128/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_128/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_128/hlo.mlirbc",
        ),
    ],
)

RESNET50_FP32_TF_224X224X3XF32_BATCH256 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH256,
    name="RESNET50_FP32_TF_224X224X3XF32_BATCH256",
    tags=["batch-256"],
    meta_model=RESNET50_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_256/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_256/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_256/hlo.mlirbc",
        ),
    ],
)

RESNET50_FP32_TF_224X224X3XF32_BATCH2048 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH2048,
    name="RESNET50_FP32_TF_224X224X3XF32_BATCH2048",
    tags=["batch-2048"],
    meta_model=RESNET50_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_2048/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_2048/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/RESNET50/batch_2048/hlo.mlirbc",
        ),
    ],
)

RESNET50_FP32_TF_224X224X3XF32_BATCHES = {
    1: RESNET50_FP32_TF_224X224X3XF32_BATCH1,
    8: RESNET50_FP32_TF_224X224X3XF32_BATCH8,
    64: RESNET50_FP32_TF_224X224X3XF32_BATCH64,
    128: RESNET50_FP32_TF_224X224X3XF32_BATCH128,
    256: RESNET50_FP32_TF_224X224X3XF32_BATCH256,
    2048: RESNET50_FP32_TF_224X224X3XF32_BATCH2048,
}

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_LARGE_FP32_TF_384XI32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH1",
    tags=["batch-1"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH16 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH16,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH16",
    tags=["batch-16"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_16/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_16/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_16/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH24 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH24,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH24",
    tags=["batch-24"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_24/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_24/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_24/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH32 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH32,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH32",
    tags=["batch-32"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_32/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_32/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_32/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH48 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH48,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH48",
    tags=["batch-48"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_48/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_48/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_48/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH64,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH64",
    tags=["batch-64"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_64/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_64/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_64/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH512 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH512,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH512",
    tags=["batch-512"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_512/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_512/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_512/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH1024 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1024,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH1024",
    tags=["batch-1024"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1024/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1024/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1024/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCH1280 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1280,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH1280",
    tags=["batch-1280"],
    meta_model=BERT_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1280/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1280/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/BERT_LARGE/batch_1280/hlo.mlirbc",
        ),
    ],
)

BERT_LARGE_FP32_TF_384XI32_BATCHES = {
    1: BERT_LARGE_FP32_TF_384XI32_BATCH1,
    16: BERT_LARGE_FP32_TF_384XI32_BATCH16,
    24: BERT_LARGE_FP32_TF_384XI32_BATCH24,
    32: BERT_LARGE_FP32_TF_384XI32_BATCH32,
    48: BERT_LARGE_FP32_TF_384XI32_BATCH48,
    64: BERT_LARGE_FP32_TF_384XI32_BATCH64,
    512: BERT_LARGE_FP32_TF_384XI32_BATCH512,
    1024: BERT_LARGE_FP32_TF_384XI32_BATCH1024,
    1280: BERT_LARGE_FP32_TF_384XI32_BATCH1280,
}

# T5-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model
# Bert-Large batch sizes used for T5-Large models.
T5_LARGE_FP32_TF_512XI32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH1,
    name="T5_LARGE_FP32_TF_512XI32_BATCH1",
    tags=["batch-1"],
    meta_model=T5_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_1/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_1/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_1/hlo.mlirbc",
        ),
    ],
)

T5_LARGE_FP32_TF_512XI32_BATCH16 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH16,
    name="T5_LARGE_FP32_TF_512XI32_BATCH16",
    tags=["batch-16"],
    meta_model=T5_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_16/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_16/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_16/hlo.mlirbc",
        ),
    ],
)

T5_LARGE_FP32_TF_512XI32_BATCH24 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH24,
    name="T5_LARGE_FP32_TF_512XI32_BATCH24",
    tags=["batch-24"],
    meta_model=T5_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_24/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_24/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_24/hlo.mlirbc",
        ),
    ],
)

T5_LARGE_FP32_TF_512XI32_BATCH32 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH32,
    name="T5_LARGE_FP32_TF_512XI32_BATCH32",
    tags=["batch-32"],
    meta_model=T5_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_32/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_32/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_32/hlo.mlirbc",
        ),
    ],
)

T5_LARGE_FP32_TF_512XI32_BATCH48 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH48,
    name="T5_LARGE_FP32_TF_512XI32_BATCH48",
    tags=["batch-48"],
    meta_model=T5_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_48/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_48/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_48/hlo.mlirbc",
        ),
    ],
)

T5_LARGE_FP32_TF_512XI32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH64,
    name="T5_LARGE_FP32_TF_512XI32_BATCH64",
    tags=["batch-64"],
    meta_model=T5_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_64/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_64/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_64/hlo.mlirbc",
        ),
    ],
)

T5_LARGE_FP32_TF_512XI32_BATCH512 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH512,
    name="T5_LARGE_FP32_TF_512XI32_BATCH512",
    tags=["batch-512"],
    meta_model=T5_LARGE_FP32_TF,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_512/hlo.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_512/tf-model.tar.gz",
        ),
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=f"{PARENT_GCS_DIR}/T5_LARGE/batch_512/hlo.mlirbc",
        ),
    ],
)

T5_LARGE_FP32_TF_512XI32_BATCHES = {
    1: T5_LARGE_FP32_TF_512XI32_BATCH1,
    16: T5_LARGE_FP32_TF_512XI32_BATCH16,
    24: T5_LARGE_FP32_TF_512XI32_BATCH24,
    32: T5_LARGE_FP32_TF_512XI32_BATCH32,
    48: T5_LARGE_FP32_TF_512XI32_BATCH48,
    64: T5_LARGE_FP32_TF_512XI32_BATCH64,
    512: T5_LARGE_FP32_TF_512XI32_BATCH512,
}

# Dictionaries.
TF_MODELS_DICT = {
    unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH1:
        RESNET50_FP32_TF_224X224X3XF32_BATCH1,
    unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH8:
        RESNET50_FP32_TF_224X224X3XF32_BATCH8,
    unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH64:
        RESNET50_FP32_TF_224X224X3XF32_BATCH64,
    unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH128:
        RESNET50_FP32_TF_224X224X3XF32_BATCH128,
    unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH256:
        RESNET50_FP32_TF_224X224X3XF32_BATCH256,
    unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH2048:
        RESNET50_FP32_TF_224X224X3XF32_BATCH2048,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1:
        BERT_LARGE_FP32_TF_384XI32_BATCH1,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH16:
        BERT_LARGE_FP32_TF_384XI32_BATCH16,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH24:
        BERT_LARGE_FP32_TF_384XI32_BATCH24,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH32:
        BERT_LARGE_FP32_TF_384XI32_BATCH32,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH48:
        BERT_LARGE_FP32_TF_384XI32_BATCH48,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH64:
        BERT_LARGE_FP32_TF_384XI32_BATCH64,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH512:
        BERT_LARGE_FP32_TF_384XI32_BATCH512,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1024:
        BERT_LARGE_FP32_TF_384XI32_BATCH1024,
    unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1280:
        BERT_LARGE_FP32_TF_384XI32_BATCH1280,
    unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH1:
        T5_LARGE_FP32_TF_512XI32_BATCH1,
    unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH16:
        T5_LARGE_FP32_TF_512XI32_BATCH16,
    unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH24:
        T5_LARGE_FP32_TF_512XI32_BATCH24,
    unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH32:
        T5_LARGE_FP32_TF_512XI32_BATCH32,
    unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH48:
        T5_LARGE_FP32_TF_512XI32_BATCH48,
    unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH64:
        T5_LARGE_FP32_TF_512XI32_BATCH64,
    unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH512:
        T5_LARGE_FP32_TF_512XI32_BATCH512,
}
