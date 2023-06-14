import itertools
import string

import data_types
import data_types_builder
import input_data_definitions
import pytorch_output_data_definitions
import unique_ids

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230401.795_1680469670"

# Meta models.
RESNET50_FP32_PT = data_types.MetaModel(
    id=unique_ids.MODEL_RESNET50_FP32_PT,
    name="RESNET50_FP32_PT",
    tags=["fp32", "cnn", "resnet"],
    framework_type=data_types.ModelFrameworkType.PYTORCH,
    source_info=
    "https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html",
    data_type=data_types.DataType.FP32,
)

RESNET50_FP16_PT = data_types.MetaModel(
    id=unique_ids.MODEL_RESNET50_FP16_PT,
    name="RESNET50_FP16_PT",
    tags=["fp16", "cnn", "resnet"],
    framework_type=data_types.ModelFrameworkType.PYTORCH,
    source_info=
    "https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html",
    data_type=data_types.DataType.FP16,
)

BERT_LARGE_FP32_PT = data_types.MetaModel(
    id=unique_ids.MODEL_BERT_LARGE_FP32_PT,
    name="BERT_LARGE_FP32_PT",
    tags=["fp32", "transformer-encoder", "bert"],
    framework_type=data_types.ModelFrameworkType.PYTORCH,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel",
    data_type=data_types.DataType.FP32,
)

BERT_LARGE_FP16_PT = data_types.MetaModel(
    id=unique_ids.MODEL_BERT_LARGE_FP16_PT,
    name="BERT_LARGE_FP16_PT",
    tags=["fp16", "transformer-encoder", "bert"],
    framework_type=data_types.ModelFrameworkType.PYTORCH,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel",
    data_type=data_types.DataType.FP16,
)

# Constants and functions help build batch templates.
BATCH_ID = lambda model_id: string.Template(model_id + "-batch${batch_size}")
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")

# Resnet50 models.
# Model implementation from https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
RESNET50_FP32_PT_3X224X224XF32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_RESNET50_FP32_PT_3X224X224XF32),
    name=BATCH_NAME("RESNET50_FP32_PT_3X224X224XF32"),
    tags=[BATCH_TAG],
    meta_model=RESNET50_FP32_PT,
    inputs=input_data_definitions.IMAGENET_APPLES_PT_3X224X224XF32_BATCHES,
    outputs=pytorch_output_data_definitions.RESNET50_FP32_PT_2048X7X7XF32_BATCHES,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_LINALG,
            source_url=string.Template(
                PARENT_GCS_DIR + "/RESNET50/batch_${batch_size}/linalg.mlir"),
        ),
    ])
RESNET50_FP32_PT_3X224X224XF32_BATCHES = data_types_builder.build_batch_models(
    template=RESNET50_FP32_PT_3X224X224XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

RESNET50_FP16_PT_3X224X224XF16_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_RESNET50_FP16_PT_3X224X224XF16),
    name=BATCH_NAME("RESNET50_FP16_PT_3X224X224XF16"),
    tags=[BATCH_TAG],
    meta_model=RESNET50_FP16_PT,
    inputs=input_data_definitions.IMAGENET_APPLES_3X224X224XF16_BATCHES,
    outputs=pytorch_output_data_definitions.RESNET50_FP16_PT_2048X7X7XF16_BATCHES,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_LINALG,
            source_url=string.Template(
                "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/RESNET50_FP16/batch_${batch_size}/linalg.mlir"),
        ),
    ])
RESNET50_FP16_PT_3X224X224XF16_BATCHES = data_types_builder.build_batch_models(
    template=RESNET50_FP16_PT_3X224X224XF16_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_LARGE_FP32_PT_384XI32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP32_PT_384XI32),
    name=BATCH_NAME("BERT_LARGE_FP32_PT_384XI32"),
    tags=[BATCH_TAG],
    meta_model=BERT_LARGE_FP32_PT,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCHES,
    outputs=pytorch_output_data_definitions.
    BERT_LARGE_FP32_PT_384X1024XF32_BATCHES,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_LINALG,
            source_url=string.Template(
                PARENT_GCS_DIR + "/BERT_LARGE/batch_${batch_size}/linalg.mlir"),
        ),
    ])
BERT_LARGE_FP32_PT_384XI32_BATCHES = data_types_builder.build_batch_models(
    template=BERT_LARGE_FP32_PT_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

BERT_LARGE_FP16_PT_384XI32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP16_PT_384XI32),
    name=BATCH_NAME("BERT_LARGE_FP16_PT_384XI32"),
    tags=[BATCH_TAG],
    meta_model=BERT_LARGE_FP16_PT,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCHES,
    outputs=pytorch_output_data_definitions.BERT_LARGE_FP16_PT_384X1024XF16_BATCHES,
    artifacts=[
        data_types.ModelArtifact(
            artifact_type=data_types.ModelArtifactType.MLIR_LINALG,
            source_url=string.Template(
                "https://storage.googleapis.com/iree-model-artifacts/pytorch/torch_models_20230522.846_1684830698/BERT_LARGE_FP16/batch_${batch_size}/linalg.mlir"),
        ),
    ])
BERT_LARGE_FP16_PT_384XI32_BATCHES = data_types_builder.build_batch_models(
    template=BERT_LARGE_FP16_PT_384XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# Collections.
PT_MODELS = list(
    itertools.chain(RESNET50_FP32_PT_3X224X224XF32_BATCHES.values(),
                    RESNET50_FP16_PT_3X224X224XF16_BATCHES.values(),
                    BERT_LARGE_FP32_PT_384XI32_BATCHES.values(),
                    BERT_LARGE_FP16_PT_384XI32_BATCHES.values()))
PT_MODELS_DICT = dict((model.id, model) for model in PT_MODELS)
