import itertools
import string

import data_types
import data_types_builder
import input_data_definitions
import tf_output_data_definitions
import unique_ids

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1685428719"

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

# Constants and functions help build batch templates.
BATCH_ID = lambda model_id: string.Template(model_id + "-batch${batch_size}")
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")

# Resnet50 models.
# Model implementation from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
RESNET50_FP32_TF_224X224X3XF32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32),
    name=BATCH_NAME("RESNET50_FP32_TF_224X224X3XF32"),
    tags=[BATCH_TAG],
    meta_model=RESNET50_FP32_TF,
    inputs=input_data_definitions.IMAGENET_APPLES_TF_224X224X3XF32_BATCHES,
    outputs=tf_output_data_definitions.RESNET50_FP32_TF_1000XF32_BATCHES,
    artifacts=[
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/RESNET50/batch_${batch_size}/hlo/inference_forward.before_optimizations.txt"
            ),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/RESNET50/batch_${batch_size}/tf-model.tar.gz"),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/RESNET50/batch_${batch_size}/stablehlo.mlirbc"),
        ),
    ])
RESNET50_FP32_TF_224X224X3XF32_BATCHES = data_types_builder.build_batch_models(
    template=RESNET50_FP32_TF_224X224X3XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32),
    name=BATCH_NAME("BERT_LARGE_FP32_TF_384XI32"),
    tags=[BATCH_TAG],
    meta_model=BERT_LARGE_FP32_TF,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCHES,
    outputs=tf_output_data_definitions.BERT_LARGE_FP32_TF_384X1024XF32_BATCHES,
    artifacts=[
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/BERT_LARGE/batch_${batch_size}/hlo/inference_forward.before_optimizations.txt"
            ),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/BERT_LARGE/batch_${batch_size}/tf-model.tar.gz"),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/BERT_LARGE/batch_${batch_size}/stablehlo.mlirbc"),
        ),
    ])
BERT_LARGE_FP32_TF_384XI32_BATCHES = data_types_builder.build_batch_models(
    template=BERT_LARGE_FP32_TF_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# T5-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/t5#transformers.TFT5Model
# Bert-Large batch sizes used for T5-Large models.
T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32),
    name=BATCH_NAME("T5_LARGE_FP32_TF_512XI32"),
    tags=[BATCH_TAG],
    meta_model=T5_LARGE_FP32_TF,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCHES,
    outputs=tf_output_data_definitions.T5_LARGE_FP32_TF_512X1024XF32_BATCHES,
    artifacts=[
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.TF_HLO_DUMP,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/T5_LARGE/batch_${batch_size}/hlo/inference_forward.before_optimizations.txt"
            ),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.TF_SAVEDMODEL_V2,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/T5_LARGE/batch_${batch_size}/tf-model.tar.gz"),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/T5_LARGE/batch_${batch_size}/stablehlo.mlirbc"),
        ),
    ])
T5_LARGE_FP32_TF_512XI32_BATCHES = data_types_builder.build_batch_models(
    template=T5_LARGE_FP32_TF_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

# Collections.
TF_MODELS = list(
    itertools.chain(RESNET50_FP32_TF_224X224X3XF32_BATCHES.values(),
                    BERT_LARGE_FP32_TF_384XI32_BATCHES.values(),
                    T5_LARGE_FP32_TF_512XI32_BATCHES.values()))
TF_MODELS_DICT = dict((model.id, model) for model in TF_MODELS)
