import data_types
import input_data_definitions
import tf_output_data_definitions
import unique_ids

from typing import List

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794"


# Resnet50 models.
# Model implementation from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
RESNET50_FP32_TF_3X224X224XF32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH1,
    name="RESNET50_FP32_TF_3X224X224XF32_BATCH1",
    tags=["fp32", "cnn", "resnet", "batch-1"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
    input_batch_size=1,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH1,
    outputs=tf_output_data_definitions.RESNET50_FP32_TF_1000XF32_BATCH1,
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

RESNET50_FP32_TF_3X224X224XF32_BATCH8 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH8,
    name="RESNET50_FP32_TF_3X224X224XF32_BATCH8",
    tags=["fp32", "cnn", "resnet", "batch-8"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
    input_batch_size=8,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH8,
    outputs=tf_output_data_definitions.RESNET50_FP32_TF_1000XF32_BATCH8,
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

RESNET50_FP32_TF_3X224X224XF32_BATCH64 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH64,
    name="RESNET50_FP32_TF_3X224X224XF32_BATCH64",
    tags=["fp32", "cnn", "resnet", "batch-64"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
    input_batch_size=64,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH64,
    outputs=tf_output_data_definitions.RESNET50_FP32_TF_1000XF32_BATCH64,
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

RESNET50_FP32_TF_3X224X224XF32_BATCH128 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH128,
    name="RESNET50_FP32_TF_3X224X224XF32_BATCH128",
    tags=["fp32", "cnn", "resnet", "batch-128"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
    input_batch_size=128,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH128,
    outputs=tf_output_data_definitions.RESNET50_FP32_TF_1000XF32_BATCH128,
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

RESNET50_FP32_TF_3X224X224XF32_BATCH256 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH256,
    name="RESNET50_FP32_TF_3X224X224XF32_BATCH256",
    tags=["fp32", "cnn", "resnet", "batch-256"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
    input_batch_size=256,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH256,
    outputs=tf_output_data_definitions.RESNET50_FP32_TF_1000XF32_BATCH256,
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

RESNET50_FP32_TF_3X224X224XF32_BATCH2048 = data_types.Model(
    id=unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH2048,
    name="RESNET50_FP32_TF_3X224X224XF32_BATCH2048",
    tags=["fp32", "cnn", "resnet", "batch-2048"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
    input_batch_size=2048,
    inputs=input_data_definitions.IMAGENET_APPLES_224X224X3XF32_BATCH2048,
    outputs=tf_output_data_definitions.RESNET50_FP32_TF_1000XF32_BATCH2048,
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


# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_LARGE_FP32_TF_384XI32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1,
    name="BERT_LARGE_FP32_TF_384XI32_BATCH1",
    tags=["fp32", "transformer-encoder", "bert", "batch-1"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=1,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH1,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH1,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-16"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=16,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH16,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH16,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-24"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=24,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH24,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH24,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-32"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=32,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH32,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH32,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-48"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=48,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH48,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH48,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-64"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=64,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH64,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH64,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-512"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=512,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH512,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH512,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-1024"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=1024,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH1024,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH1024,
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
    tags=["fp32", "transformer-encoder", "bert", "batch-1280"],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel",
    input_batch_size=1280,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCH1280,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCH1280,
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

# T5-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model
# Bert-Large batch sizes used for T5-Large models.
T5_LARGE_FP32_TF_512XI32_BATCH1 = data_types.Model(
    id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH1,
    name="T5_LARGE_FP32_TF_512XI32_BATCH1",
    tags=[
        "fp32", "transformer-encoder", "transformer-decoder", "t5", "batch-1"
    ],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    input_batch_size=1,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH1,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCH1,
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
    tags=[
        "fp32", "transformer-encoder", "transformer-decoder", "t5", "batch-16"
    ],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    input_batch_size=16,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH16,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCH16,
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
    tags=[
        "fp32", "transformer-encoder", "transformer-decoder", "t5", "batch-24"
    ],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    input_batch_size=24,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH24,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCH24,
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
    tags=[
        "fp32", "transformer-encoder", "transformer-decoder", "t5", "batch-32"
    ],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    input_batch_size=32,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH32,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCH32,
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
    tags=[
        "fp32", "transformer-encoder", "transformer-decoder", "t5", "batch-48"
    ],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    input_batch_size=48,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH48,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCH48,
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
    tags=[
        "fp32", "transformer-encoder", "transformer-decoder", "t5", "batch-64"
    ],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    input_batch_size=64,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH64,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCH64,
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
    tags=[
        "fp32", "transformer-encoder", "transformer-decoder", "t5", "batch-512"
    ],
    framework_type=data_types.ModelFrameworkType.TENSORFLOW_V2,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model",
    input_batch_size=512,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCH512,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCH512,
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

# Dictionaries.
TF_MODELS_DICT = {
    unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH1:
        RESNET50_FP32_TF_3X224X224XF32_BATCH1,
    unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH8:
        RESNET50_FP32_TF_3X224X224XF32_BATCH8,
    unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH64:
        RESNET50_FP32_TF_3X224X224XF32_BATCH64,
    unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH128:
        RESNET50_FP32_TF_3X224X224XF32_BATCH128,
    unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH256:
        RESNET50_FP32_TF_3X224X224XF32_BATCH256,
    unique_ids.MODEL_RESNET50_FP32_TF_3X224X224XF32_BATCH2048:
        RESNET50_FP32_TF_3X224X224XF32_BATCH2048,
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
