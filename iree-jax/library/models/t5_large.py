from transformers import AutoTokenizer, FlaxT5Model
import jax.numpy as jnp

# We use a maximum sequence length of 512 since this is the default used in the T5 config.
_SEQUENCE_LENGTH = 512


class T5Large():

  def __init__(self, dtype=jnp.float32):
    self.model = FlaxT5Model.from_pretrained("t5-large", return_dict=True, dtype=dtype)
    if dtype == jnp.float16:
      self.model.params = self.model.to_fp16(self.model.params)
    elif dtype == jnp.bfloat16:
      self.model.params = self.model.to_bf16(self.model.params)

  def generate_inputs(self, batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    tokenization_kwargs = {
        "pad_to_multiple_of": _SEQUENCE_LENGTH,
        "padding": True,
        "return_tensors": "jax",
    }

    text = "Studies have been shown that owning a dog is good for you"
    batched_text = [text] * batch_size
    encoded_input_ids = tokenizer(batched_text, **tokenization_kwargs).input_ids

    text = "Studies show that"
    batched_text = [text] * batch_size
    decoder_input_ids = tokenizer(batched_text, **tokenization_kwargs).input_ids
    # The HuggingFace documentation reports that _shift_right() exists for
    # `FlaxT5Model` but we get an attribute does not exist error. Disabling for now.
    # decoder_input_ids = self.model._shift_right(decoder_input_ids)

    return (encoded_input_ids, decoder_input_ids)

  def forward(self, input_ids, decoder_input_ids):
    return self.model(input_ids, decoder_input_ids=decoder_input_ids)[0]
