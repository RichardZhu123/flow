import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

class GRUModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.gru = nn.GRU(input_size=obs_space.shape[0],
                          hidden_size=5,
                          batch_first=True)
        self.fc = nn.Linear(5, num_outputs)
        self.value_branch = nn.Linear(5, 1)
        self._value = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        x, _ = self.gru(input_dict["obs"].float())
        x = torch.tanh(x)
        self._value = self.value_branch(x).squeeze(1)
        return self.fc(x), state

    @override(ModelV2)
    def value_function(self):
        return self._value.squeeze(1)
