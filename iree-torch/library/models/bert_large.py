import torch
from transformers import BertTokenizer, BertModel

_SEQUENCE_LENGTH = 384


# We use the Bert-Large variant listed in MLPerf here: https://github.com/mlcommons/inference/tree/master/language/bert
# Where `max_seq_length` is 384.
class BertLarge(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-large-uncased")

    def generate_inputs(self, batch_size=1):
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        input_text = ["a photo of a cat"] * batch_size
        inputs = tokenizer(text=input_text,
                           padding="max_length",
                           max_length=_SEQUENCE_LENGTH,
                           return_tensors="pt")

        return (inputs["input_ids"], inputs["attention_mask"])

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]
