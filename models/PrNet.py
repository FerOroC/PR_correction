# model structure
import torch
from torch import nn

torch.set_default_tensor_type(torch.DoubleTensor)


# %% ********************* Mlp Feature Extractor  *********************
# Input shape:  batch_size * Prn_size * input_size
# Output PrM bias shape: batch_size * Prn_size * 1
class MlpFeatureExtractor(nn.Module):
    """The MLP encoder for debiasing."""

    def __init__(self, input_size_debiasing, num_hiddens_debiasing,
                 num_debiasing_layers, dropout=0, **kwargs):
        super(MlpFeatureExtractor, self).__init__(**kwargs)
        # %% Debiasing layer
        self.num_debiasing_layers = num_debiasing_layers
        self.debiasing_mlp_0 = nn.Sequential(nn.Linear(input_size_debiasing, num_hiddens_debiasing),
                                             nn.ReLU(),
                                             # Add a dropout layer after the first fully connected layer
                                             nn.Dropout(dropout))
        if num_debiasing_layers >= 2:
            self.debiasing_mlp = nn.ModuleList([nn.Sequential(nn.Linear(num_hiddens_debiasing, num_hiddens_debiasing),
                                                              nn.ReLU(),
                                                              # Add a dropout layer after each fully connected layer
                                                              nn.Dropout(dropout)) for i in
                                                range(num_debiasing_layers - 1)])
            # Output fully connected layer
        self.debiasing_mlp_output = nn.Linear(num_hiddens_debiasing, 1)

    def forward(self, inputs, *args):
        X = inputs

        # %%Debiasing layer
        for i in range(self.num_debiasing_layers):  # i indexes the layers
            if i == 0:  # the 1st layer
                h_temp = self.debiasing_mlp_0(X)
            else:  # the higher layer
                h = h_temp
                h_temp = self.debiasing_mlp[i - 1](h)
        # Shape of prm_error_bias: ('batch_size', 'Prn_size', 1)
        prm_error_bias = self.debiasing_mlp_output(h_temp)

        # `output` shape: ('batch_size', 'Prn_size', 1)
        return prm_error_bias


# %%  ********************* PrNet *********************
# @save
class PrNet(nn.Module):
    """The base class for PrNet."""

    def __init__(self, debiasing_layer, **kwargs):
        super(PrNet, self).__init__(**kwargs)
        self.debiasing_layer = debiasing_layer

    def forward(self, feature_inputs, *args):
        inputs_bias = self.debiasing_layer(feature_inputs, *args)
        # `output` shape: ('batch_size', 'Prn_size', 1)
        return inputs_bias