import jax_model_definitions
import tf_model_definitions

MODEL_DICT = (jax_model_definitions.JAX_MODELS_DICT |
              tf_model_definitions.TF_MODELS_DICT)
