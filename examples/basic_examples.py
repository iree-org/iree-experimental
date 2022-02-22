import torch
from shark_runner import shark_inference


class Mlp1LayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.fc0 = torch.nn.Linear(3, 5)
        self.tanh0 = torch.nn.Tanh()

    def forward(self, x):
        return self.tanh0(self.fc0(x))


input = torch.rand(5, 3).to(torch.float32)

result_static = shark_inference(Mlp1LayerModule(), input, device="cpu", dynamic=False)
result_dynamic = shark_inference(Mlp1LayerModule(), input, device="cpu", dynamic=True)


x = (result_dynamic == result_static).all()

# True
print(x)
