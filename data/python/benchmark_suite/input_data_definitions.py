import data_types
import unique_ids

IMAGENET_APPLES_1X224X224X3XF32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_1X224X224X3XF32,
    name="Imagenet_Apples_1x224x224x3xf32",
    tags=["imagenet", "input-data"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    source_url="https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_1/input_0.npy",
    source_info="https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_dimensions="1x224x224xf32",
)

IMAGENET_APPLES_8X224X224X3XF32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_8X224X224X3XF32,
    name="Imagenet_Apples_8x224x224x3xf32",
    tags=["imagenet", "input-data"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    source_url="https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_8/input_0.npy",
    source_info="https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_dimensions="8x224x224xf32",
)

IMAGENET_APPLES_64X224X224X3XF32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_64X224X224X3XF32,
    name="Imagenet_Apples_64x224x224x3xf32",
    tags=["imagenet", "input-data"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    source_url="https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_64/input_0.npy",
    source_info="https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_dimensions="64x224x224xf32",
)

IMAGENET_APPLES_128X224X224X3XF32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_128X224X224X3XF32,
    name="Imagenet_Apples_128x224x224x3xf32",
    tags=["imagenet", "input-data"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    source_url="https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_128/input_0.npy",
    source_info="https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_dimensions="128x224x224xf32",
)

IMAGENET_APPLES_256X224X224X3XF32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_256X224X224X3XF32,
    name="Imagenet_Apples_256x224x224x3xf32",
    tags=["imagenet", "input-data"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    source_url="https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_256/input_0.npy",
    source_info="https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_dimensions="256x224x224xf32",
)

IMAGENET_APPLES_2048X224X224X3XF32 = data_types.ModelData(
    id=unique_ids.INPUT_DATA_IMAGENET_APPLES_2048X224X224X3XF32,
    name="Imagenet_Apples_2048x224x224x3xf32",
    tags=["imagenet", "input-data"],
    data_format=data_types.DataFormat.NUMPY_NPY,
    source_url="https://storage.googleapis.com/iree-model-artifacts/tensorflow/tf_models_2.12.0_1681767794/RESNET50/batch_2048/input_0.npy",
    source_info="https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_dimensions="2048x224x224xf32",
)
