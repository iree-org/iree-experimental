import data_types
import input_data_definitions
import output_data_definitions
import unique_ids

# Resnet50 models.
# Model implementation from https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50

RESNET50_1X3X224X224XF32_FP32_TF = data_types.Model(
   id=unique_ids.MODEL_RESNET50_1X3X224X224XF32_FP32_TF,
   name="ResNet50_1X3X224X224XF32_FP32_TF",
   tags=["fp32", "cnn", "batch-1"],
   framework_type=data_types.FrameworkType.TENSORFLOW_V2,
   source_info="https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
   input_batch_size=1,
   inputs=[input_data_definitions.IMAGENET_APPLES_1X224X224X3XF32],
   expected_outputs=[output_data_definitions.RESNET50_1X3X224X224XF32_FP32_TF_OUTPUT_0],
)

RESNET50_8X3X224X224XF32_FP32_TF = data_types.Model(
   id=unique_ids.MODEL_RESNET50_8X3X224X224XF32_FP32_TF,
   name="ResNet50_8X3X224X224XF32_FP32_TF",
   tags=["fp32", "cnn", "batch-8"],
   framework_type=data_types.FrameworkType.TENSORFLOW_V2,
   source_info="https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
   input_batch_size=8,
   inputs=[input_data_definitions.IMAGENET_APPLES_8X224X224X3XF32],
   expected_outputs=[output_data_definitions.RESNET50_8X3X224X224XF32_FP32_TF_OUTPUT_0],
)

RESNET50_64X3X224X224XF32_FP32_TF = data_types.Model(
   id=unique_ids.MODEL_RESNET50_64X3X224X224XF32_FP32_TF,
   name="ResNet50_64X3X224X224XF32_FP32_TF",
   tags=["fp32", "cnn", "batch-64"],
   framework_type=data_types.FrameworkType.TENSORFLOW_V2,
   source_info="https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
   input_batch_size=64,
   inputs=[input_data_definitions.IMAGENET_APPLES_64X224X224X3XF32],
   expected_outputs=[output_data_definitions.RESNET50_64X3X224X224XF32_FP32_TF_OUTPUT_0],
)

RESNET50_128X3X224X224XF32_FP32_TF = data_types.Model(
   id=unique_ids.MODEL_RESNET50_128X3X224X224XF32_FP32_TF,
   name="ResNet50_128X3X224X224XF32_FP32_TF",
   tags=["fp32", "cnn", "batch-128"],
   framework_type=data_types.FrameworkType.TENSORFLOW_V2,
   source_info="https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
   input_batch_size=128,
   inputs=[input_data_definitions.IMAGENET_APPLES_128X224X224X3XF32],
   expected_outputs=[output_data_definitions.RESNET50_128X3X224X224XF32_FP32_TF_OUTPUT_0],
)

RESNET50_256X3X224X224XF32_FP32_TF = data_types.Model(
   id=unique_ids.MODEL_RESNET50_256X3X224X224XF32_FP32_TF,
   name="ResNet50_256X3X224X224XF32_FP32_TF",
   tags=["fp32", "cnn", "batch-256"],
   framework_type=data_types.FrameworkType.TENSORFLOW_V2,
   source_info="https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
   input_batch_size=256,
   inputs=[input_data_definitions.IMAGENET_APPLES_256X224X224X3XF32],
   expected_outputs=[output_data_definitions.RESNET50_256X3X224X224XF32_FP32_TF_OUTPUT_0],
)

RESNET50_2048X3X224X224XF32_FP32_TF = data_types.Model(
   id=unique_ids.MODEL_RESNET50_2048X3X224X224XF32_FP32_TF,
   name="ResNet50_2048X3X224X224XF32_FP32_TF",
   tags=["fp32", "cnn", "batch-2048"],
   framework_type=data_types.FrameworkType.TENSORFLOW_V2,
   source_info="https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50",
   input_batch_size=2048,
   inputs=[input_data_definitions.IMAGENET_APPLES_2048X224X224X3XF32],
   expected_outputs=[output_data_definitions.RESNET50_2048X3X224X224XF32_FP32_TF_OUTPUT_0],
)

TF_MODELS_DICT = {
    unique_ids.MODEL_RESNET50_1X3X224X224XF32_FP32_TF: RESNET50_1X3X224X224XF32_FP32_TF,
    unique_ids.MODEL_RESNET50_8X3X224X224XF32_FP32_TF: RESNET50_8X3X224X224XF32_FP32_TF,
    unique_ids.MODEL_RESNET50_64X3X224X224XF32_FP32_TF: RESNET50_64X3X224X224XF32_FP32_TF,
    unique_ids.MODEL_RESNET50_128X3X224X224XF32_FP32_TF: RESNET50_128X3X224X224XF32_FP32_TF,
    unique_ids.MODEL_RESNET50_256X3X224X224XF32_FP32_TF: RESNET50_256X3X224X224XF32_FP32_TF,
    unique_ids.MODEL_RESNET50_2048X3X224X224XF32_FP32_TF: RESNET50_2048X3X224X224XF32_FP32_TF,
}
