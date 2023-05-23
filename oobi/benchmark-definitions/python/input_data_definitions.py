import data_types
import unique_ids

IMAGENET_APPLES_224X224X3XF32_BATCH1 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_224X224X3XF32_BATCH1,
    name="IMAGENET_APPLES_224X224X3XF32_BATCH1",
    tags=["input-data", "imagenet", "batch-1"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH1,
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_names=["serving_default_inputs"],
    tensor_dimensions=["1x224x224x3xf32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_1/input_0.npy"
    ],
)

IMAGENET_APPLES_224X224X3XF32_BATCH8 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_224X224X3XF32_BATCH8,
    name="IMAGENET_APPLES_224X224X3XF32_BATCH8",
    tags=["input-data", "imagenet", "batch-8"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH8,
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_names=["serving_default_inputs"],
    tensor_dimensions=["8x224x224x3xf32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_8/input_0.npy"
    ],
)

IMAGENET_APPLES_224X224X3XF32_BATCH64 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_224X224X3XF32_BATCH64,
    name="IMAGENET_APPLES_224X224X3XF32_BATCH64",
    tags=["input-data", "imagenet", "batch-64"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH64,
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_names=["serving_default_inputs"],
    tensor_dimensions=["64x224x224x3xf32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_64/input_0.npy"
    ],
)

IMAGENET_APPLES_224X224X3XF32_BATCH128 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_224X224X3XF32_BATCH128,
    name="IMAGENET_APPLES_224X224X3XF32_BATCH128",
    tags=["input-data", "imagenet", "batch-128"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH128,
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_names=["serving_default_inputs"],
    tensor_dimensions=["128x224x224x3xf32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_128/input_0.npy"
    ],
)

IMAGENET_APPLES_224X224X3XF32_BATCH256 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_224X224X3XF32_BATCH256,
    name="IMAGENET_APPLES_224X224X3XF32_BATCH256",
    tags=["input-data", "imagenet", "batch-256"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH256,
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_names=["serving_default_inputs"],
    tensor_dimensions=["256x224x224x3xf32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_256/input_0.npy"
    ],
)

IMAGENET_APPLES_224X224X3XF32_BATCH2048 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_224X224X3XF32_BATCH2048,
    name="IMAGENET_APPLES_224X224X3XF32_BATCH2048",
    tags=["input-data", "imagenet", "batch-2048"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_RESNET50_FP32_TF_224X224X3XF32_BATCH2048,
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_names=["serving_default_inputs"],
    tensor_dimensions=["2048x224x224x3xf32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_2048/input_0.npy"
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH1 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH1,
    name="BERT_LARGE_SEQLEN384_I32_BATCH1",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["1x384xi32", "1x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_1/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_1/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH16 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH16,
    name="BERT_LARGE_SEQLEN384_I32_BATCH16",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH16,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["16x384xi32", "16x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_16/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_16/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH24 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH24,
    name="BERT_LARGE_SEQLEN384_I32_BATCH24",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH24,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["24x384xi32", "24x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_24/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_24/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH32,
    name="BERT_LARGE_SEQLEN384_I32_BATCH32",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH32,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["32x384xi32", "32x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_32/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_32/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH48 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH48,
    name="BERT_LARGE_SEQLEN384_I32_BATCH48",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH48,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["48x384xi32", "48x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_48/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_48/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH64 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH64,
    name="BERT_LARGE_SEQLEN384_I32_BATCH64",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH64,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["64x384xi32", "64x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_64/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_64/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH512 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH512,
    name="BERT_LARGE_SEQLEN384_I32_BATCH512",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH512,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["512x384xi32", "512x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_512/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_512/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH1024 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH1024,
    name="BERT_LARGE_SEQLEN384_I32_BATCH1024",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1024,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["1024x384xi32", "1024x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_1024/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_1024/input_1.npy",
    ],
)

BERT_LARGE_SEQLEN384_I32_BATCH1280 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_BERT_LARGE_SEQLEN384_I32_BATCH1280,
    name="BERT_LARGE_SEQLEN384_I32_BATCH1280",
    tags=["input-data", "seqlen384"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_BERT_LARGE_FP32_TF_384XI32_BATCH1280,
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=[
        "serving_default_input_ids", "serving_default_attention_mask"
    ],
    tensor_dimensions=["1280x384xi32", "1280x384xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_1280/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/BERT_LARGE/batch_1280/input_1.npy",
    ],
)

T5_LARGE_SEQLEN512_I32_BATCH1 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_T5_LARGE_SEQLEN512_I32_BATCH1,
    name="T5_LARGE_SEQLEN512_I32_BATCH1",
    tags=["input-data", "seqlen512"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH1,
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=[
        "serving_default_input_ids", "serving_default_decoder_input_ids"
    ],
    tensor_dimensions=["1x512xi32", "1x512xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_1/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_1/input_1.npy",
    ],
)

T5_LARGE_SEQLEN512_I32_BATCH16 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_T5_LARGE_SEQLEN512_I32_BATCH16,
    name="T5_LARGE_SEQLEN512_I32_BATCH16",
    tags=["input-data", "seqlen512"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH16,
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=[
        "serving_default_input_ids", "serving_default_decoder_input_ids"
    ],
    tensor_dimensions=["16x512xi32", "16x512xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_16/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_16/input_1.npy",
    ],
)

T5_LARGE_SEQLEN512_I32_BATCH24 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_T5_LARGE_SEQLEN512_I32_BATCH24,
    name="T5_LARGE_SEQLEN512_I32_BATCH24",
    tags=["input-data", "seqlen512"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH24,
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=[
        "serving_default_input_ids", "serving_default_decoder_input_ids"
    ],
    tensor_dimensions=["24x512xi32", "24x512xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_24/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_24/input_1.npy",
    ],
)

T5_LARGE_SEQLEN512_I32_BATCH32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_T5_LARGE_SEQLEN512_I32_BATCH32,
    name="T5_LARGE_SEQLEN512_I32_BATCH32",
    tags=["input-data", "seqlen512"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH32,
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=[
        "serving_default_input_ids", "serving_default_decoder_input_ids"
    ],
    tensor_dimensions=["32x512xi32", "32x512xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_32/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_32/input_1.npy",
    ],
)

T5_LARGE_SEQLEN512_I32_BATCH48 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_T5_LARGE_SEQLEN512_I32_BATCH48,
    name="T5_LARGE_SEQLEN512_I32_BATCH48",
    tags=["input-data", "seqlen512"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH48,
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=[
        "serving_default_input_ids", "serving_default_decoder_input_ids"
    ],
    tensor_dimensions=["48x512xi32", "48x512xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_48/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_48/input_1.npy",
    ],
)

T5_LARGE_SEQLEN512_I32_BATCH64 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_T5_LARGE_SEQLEN512_I32_BATCH64,
    name="T5_LARGE_SEQLEN512_I32_BATCH64",
    tags=["input-data", "seqlen512"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH64,
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=[
        "serving_default_input_ids", "serving_default_decoder_input_ids"
    ],
    tensor_dimensions=["64x512xi32", "64x512xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_64/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_64/input_1.npy",
    ],
)

T5_LARGE_SEQLEN512_I32_BATCH512 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_T5_LARGE_SEQLEN512_I32_BATCH512,
    name="T5_LARGE_SEQLEN512_I32_BATCH512",
    tags=["input-data", "seqlen512"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=unique_ids.MODEL_T5_LARGE_FP32_TF_512XI32_BATCH512,
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=[
        "serving_default_input_ids", "serving_default_decoder_input_ids"
    ],
    tensor_dimensions=["512x512xi32", "512x512xi32"],
    source_url=[
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_512/input_0.npy",
        "https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/T5_LARGE/batch_512/input_1.npy",
    ],
)
