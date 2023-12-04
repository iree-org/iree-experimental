import jax_model_definitions
import tf_model_definitions

MODEL_DICT = (jax_model_definitions.JAX_MODELS_DICT |
              tf_model_definitions.TF_MODELS_DICT)


def get_model_definition(unique_id: str):
  if unique_id not in MODEL_DICT:
    id_list = '\n  '.join(MODEL_DICT.keys())
    raise ValueError(f"Id {unique_id} does not exist in model suite. Expected "
                     f"one of:\n  {id_list}")

  return MODEL_DICT[unique_id]
