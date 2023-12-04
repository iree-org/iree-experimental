import pathlib
import sys
import tensorflow as tf
from transformers import AutoTokenizer, TFT5Model

# We use a maximum sequence length of 512 since this is the default used in the T5 config.
_SEQUENCE_LENGTH = 512


class T5Large(tf.Module):

    def __init__(self):
        super().__init__()
        self.model = TFT5Model.from_pretrained("t5-large", return_dict=True)

    def generate_inputs(self, batch_size=1):
        tokenizer = AutoTokenizer.from_pretrained("t5-large")
        tokenization_kwargs = {
            "pad_to_multiple_of": _SEQUENCE_LENGTH,
            "padding": True,
            "return_tensors": "tf",
        }

        text = "Studies have been shown that owning a dog is good for you"
        batched_text = [text] * batch_size
        encoded_input_ids = tokenizer(batched_text, **tokenization_kwargs).input_ids

        text = "Studies show that"
        batched_text = [text] * batch_size
        decoder_input_ids = tokenizer(batched_text, **tokenization_kwargs).input_ids
        decoder_input_ids = self.model._shift_right(decoder_input_ids)

        return (encoded_input_ids, decoder_input_ids)

    @tf.function(jit_compile=True)
    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids, decoder_input_ids=decoder_input_ids)[0]
