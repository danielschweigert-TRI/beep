"""
Module for classification models.

"""
import os
import numpy as np
import torch
import torch.nn as nn

from beep import MODEL_DIR


class CycleClassificationModel(nn.Module):
    """
    Neural network topology to classify cycle type from up to the first 3000 data points of a cycle, if available.

    1-layer LSTM, 100 nodes, classifier

    input: (n_records, n_data_points, {current, voltage})
    output: (n_records, n_cycle_types) : probabilities by cycle_type

    """

    def __init__(self, input_size, hidden_size, output_size, device):
        super(CycleClassificationModel, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size + 1, self.output_size)

    def forward(self, input):
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t[:, 0], (h_t, c_t))

        length_tensor = torch.tensor([input.size(1)] * input.size(0), dtype=torch.float32).to(self.device).view((-1, 1))
        rnn_output = torch.cat((h_t, length_tensor), 1)
        output = self.linear(rnn_output)
        return output


class CycleClassifier:
    """
    Formats incoming cycler raw data and applies classification model to determine cycle type.
    """

    MODEL_FILE_PATH = os.path.join(MODEL_DIR, 'classification/model_201_1189.torch')
    CYCLE_TYPES = ['exposure', 'reset', 'hppc', 'rpt_0.2C', 'rpt_1C', 'rpt_2C']
    
    def __init__(self):
        dev = torch.device('cpu')
        self.model = CycleClassificationModel(2, 100, 6, dev)
        self.model.load_state_dict(torch.load(self.MODEL_FILE_PATH, map_location=dev))
        self.model.eval()

    def apply_cycle_type_classification(self, cycle_data):
        """
        Applies classification and returns column of cycle_types.

        Args:
            cycle_data (pandas.DataFrame):          raw cycler data frame

        Returns:
            list:                                   cycle types

        """

        n_input_data_points = 3000

        def format_input(cycle_num):

            df_sub = cycle_data[cycle_data['cycle_index'] == cycle_num]
            voltage = df_sub['voltage'].values
            current = df_sub['current'].values
            n = min(len(voltage), n_input_data_points)
            x = [[voltage[i], current[i]] for i in range(n)]

            x = np.array(x)
            x = x.reshape((1, x.shape[0], x.shape[1]))
            x = torch.tensor(x, dtype=torch.float32)

            return x

        cycle_nums = set(cycle_data['cycle_index'])
        cycle_type = list()

        for cn in cycle_nums:
            x_c = format_input(cn)
            y = self.model(x_c).detach().numpy()[0]
            ct = self.CYCLE_TYPES[np.argmax(y)]
            cycle_type.append(ct)

        return cycle_type
