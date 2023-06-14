import string

import data_types
import data_types_builder
import unique_ids

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752"

# Constants and functions help build batch templates.
BATCH_ID = lambda data_id: string.Template(data_id + "-batch${batch_size}")
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")
BATCH_MODEL_ID = lambda model_id: string.Template(model_id +
                                                  "-batch${batch_size}")
BATCH_TENSOR_DIMS = lambda dims: string.Template("${batch_size}x" + dims)

# ResNet50 Outputs.
RESNET50_FP32_JAX_2048X7X7XF32_BATCH_TEMPLATE = data_types_builder.ModelDataTemplate(
    id=BATCH_ID(unique_ids.OUTPUT_DATA_RESNET50_FP32_JAX_2048X7X7XF32),
    name=BATCH_NAME("RESNET50_FP32_JAX_2048X7X7XF32"),
    tags=["output-data", BATCH_TAG],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=BATCH_MODEL_ID(unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32),
    source_info="",
    tensor_names=["output_0"],
    tensor_dimensions=[BATCH_TENSOR_DIMS("2048x7x7xf32")],
    source_url=[
        string.Template(PARENT_GCS_DIR +
                        "/RESNET50/batch_${batch_size}/output_0.npy")
    ])
RESNET50_FP32_JAX_2048X7X7XF32_BATCHES = data_types_builder.build_batch_model_data(
    template=RESNET50_FP32_JAX_2048X7X7XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# Bert-Large Outputs.
BERT_LARGE_FP32_JAX_384X1024XF32_BATCH_TEMPLATE = data_types_builder.ModelDataTemplate(
    id=BATCH_ID(unique_ids.OUTPUT_DATA_BERT_LARGE_FP32_JAX_384X1024XF32),
    name=BATCH_NAME("BERT_LARGE_FP32_JAX_384X1024XF32"),
    tags=["output-data", BATCH_TAG],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=BATCH_MODEL_ID(unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32),
    source_info="",
    tensor_names=["output_0"],
    tensor_dimensions=[BATCH_TENSOR_DIMS("384x1024xf32")],
    source_url=[
        string.Template(PARENT_GCS_DIR +
                        "/BERT_LARGE/batch_${batch_size}/output_0.npy")
    ])
BERT_LARGE_FP32_JAX_384X1024XF32_BATCHES = data_types_builder.build_batch_model_data(
    template=BERT_LARGE_FP32_JAX_384X1024XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# T5-Large Outputs.
T5_LARGE_FP32_JAX_512X1024XF32_BATCH_TEMPLATE = data_types_builder.ModelDataTemplate(
    id=BATCH_ID(unique_ids.OUTPUT_DATA_T5_LARGE_FP32_JAX_512X1024XF32),
    name=BATCH_NAME("T5_LARGE_FP32_JAX_512X1024XF32"),
    tags=["output-data", BATCH_TAG],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=BATCH_MODEL_ID(unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32),
    source_info="",
    tensor_names=["output_0"],
    tensor_dimensions=[BATCH_TENSOR_DIMS("512x1024xf32")],
    source_url=[
        string.Template(PARENT_GCS_DIR +
                        "/T5_LARGE/batch_${batch_size}/output_0.npy")
    ])
T5_LARGE_FP32_JAX_512X1024XF32_BATCHES = data_types_builder.build_batch_model_data(
    template=T5_LARGE_FP32_JAX_512X1024XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])
