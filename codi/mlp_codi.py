"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import yaml

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from codi.confidence_discriminator import ConfidenceDiscriminator


class MLPCoDi(keras.Model, ConfidenceDiscriminator):
    r"""
    A model that discriminates between reliable and unreliable datapoints

    Parameters:
        name: string
        yaml_path: path of yaml containing model params.

    Attributes:
        layer_list : list of layers used in the model
        hyper : dictionnary of parameters in yaml file
        dp : dropout probability
    """
    def __init__(self, name='name', yaml_path='mlp_codi.yaml'):
        super(MLPCoDi, self).__init__(name=name)

        self.hyper = self.load_yaml(yaml_path)
        self.dp = None
        self.layer_list = []

    def load_model(self):
        """
        Initializes the layers with the params stored in the yaml file.
        :return: None
        """
        for h in range(0, len(self.hyper['layers'])):
            self.layer_list.append(layers.Dense(self.hyper['layers'][h]['n_neurons'],
                                                activation=self.hyper['layers'][h]['activation'],
                                                name='dense_{}'.format(h),
                                                kernel_regularizer=regularizers.l2(self.hyper['regularization_factor'])
                                                ))
        if self.hyper['use_dp']:
            self.dp = layers.Dropout(self.hyper['dp'], name='dropout')

    def load_yaml(self, path):
        """
        Return dict of yaml parameters
        """
        with open(path) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def call(self, inputs):
        """
        Computes the forward propagation.
        :return: None
        """
        x = self.layer_list[0](inputs)
        x = self.dp(x)
        for i in range(1, len(self.layer_list)-1):
            x = self.layer_list[i](x)
            if self.hyper['use_dp'][i-1]:
                x = self.dp(x)
        return self.layer_list[-1](x)
