import torch
from transformers import T5Tokenizer, T5Model

# We use a maximum sequence length of 512 since this is the default used in the T5 config.
_SEQUENCE_LENGTH = 512


class T5Large(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = T5Model.from_pretrained("t5-large")

    def generate_inputs(self, batch_size=1):
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        tokenization_kwargs = {
            "pad_to_multiple_of": _SEQUENCE_LENGTH,
            "padding": True,
            "return_tensors": "pt",
        }
        encoder_text = [
            "Studies have been shown that owning a dog is good for you"
        ] * batch_size
        encoder_inputs = tokenizer(encoder_text, **tokenization_kwargs)

        decoder_text = ["Studies show that"] * batch_size
        decoder_inputs = tokenizer(decoder_text, **tokenization_kwargs)
        decoder_input_ids = self.model._shift_right(decoder_inputs.input_ids)

        return (encoder_inputs.input_ids, decoder_input_ids)

    def forward(self, encoder_input_ids, decoder_input_ids):
        return self.model(encoder_input_ids,
                          decoder_input_ids=decoder_input_ids)[0]
