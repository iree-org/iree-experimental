import pathlib
import sys
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

_SEQUENCE_LENGTH = 384


# We use the TF2 version of BertLarge available in HuggingFace Transformers.
# The MLPerf version is TF1 which can be generated using these instructions: https://gist.github.com/mariecwhite/e61ccebd979d98d097946ac7725bcc29
class BertLarge(tf.Module):

    def __init__(self):
        super().__init__()
        self.model = TFBertModel.from_pretrained("bert-large-uncased")

    def generate_inputs(self, batch_size=1):
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        input_text = ["a photo of a cat"] * batch_size
        inputs = tokenizer(text=input_text,
                           padding="max_length",
                           max_length=_SEQUENCE_LENGTH,
                           return_tensors="tf")

        return (inputs["input_ids"], inputs["attention_mask"])
   
    @tf.function(jit_compile=True)
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask, training=False)[0]
