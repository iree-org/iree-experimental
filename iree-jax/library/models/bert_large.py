import jax

from functools import partial
from transformers import BertTokenizer, FlaxBertModel

_SEQUENCE_LENGTH = 384


class BertLarge():

  def __init__(self):
    self.model = FlaxBertModel.from_pretrained("bert-large-uncased")

  def generate_inputs(self, batch_size=1):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    input_text = ["a photo of a cat"] * batch_size
    inputs = tokenizer(text=input_text,
                       padding="max_length",
                       max_length=_SEQUENCE_LENGTH,
                       return_tensors="jax")

    return (inputs["input_ids"], inputs["attention_mask"])

  def forward(self, input_ids, attention_mask, backend="gpu"):
    return self.model(input_ids, attention_mask)
