import os
import torch
import numpy as np
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

seqlens = [64, 128, 256, 512]
model = AutoModel.from_pretrained(
    'microsoft/MiniLM-L12-H384-uncased', torchscript=True, return_dict=False)

tok = AutoTokenizer.from_pretrained(
    'microsoft/MiniLM-L12-H384-uncased')

for seqlen in seqlens:
    sample_text = "Foo bar, stakes too high"
    sample_input = tok(sample_text, padding="max_length", max_length=seqlen)
    traced_model = torch.jit.trace(
        model,
        (
            torch.tensor([sample_input['input_ids']]),
            torch.tensor([sample_input['attention_mask']]),
        )
    )
    mlprogram = ct.convert(
        traced_model,
        convert_to='mlprogram',
        inputs=[
            ct.TensorType('input_ids', shape=(1, seqlen), dtype=np.int64),
            ct.TensorType('attention_mask', shape=(1, seqlen), dtype=np.int64),
        ])
    mlpackage_path = f"./model_{seqlen}.mlpackage"
    mlprogram.save(mlpackage_path)
