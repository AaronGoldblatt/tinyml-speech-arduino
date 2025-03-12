import torch
import torch.nn as nn
import torch.nn.functional as F


# Define tiny_conv model
class Reshape(nn.Module):
    def __init__(self, output_shape):
        super(Reshape, self).__init__()
        self.output_shape = output_shape

    def __repr__(self):
        s = super().__repr__()
        s = f'{s[:-1]}output_shape={tuple(self.output_shape)})'
        return s

    def forward(self, x):
        reshaped_input = torch.reshape(x, self.output_shape)
        return reshaped_input


class TinyConv(nn.Module):
    def __init__(self, model_settings, n_input=1, n_output=4):
        super(TinyConv, self).__init__()
        first_filter_width = 8
        first_filter_height = 10
        first_filter_count = 8
        first_conv_stride_x = 2
        first_conv_stride_y = 2
        self.model_settings = model_settings

        input_frequency_size = self.model_settings['fingerprint_width']
        input_time_size = self.model_settings['spectrogram_length']
        W = input_frequency_size
        H = input_time_size
        C = 1

        # Reshape layer
        self.conv_reshape = Reshape([-1, C, H, W])

        # Conv2d layer
        self.conv = nn.Conv2d(in_channels=n_input,
                              out_channels=first_filter_count,
                              kernel_size=
                              (first_filter_height, first_filter_width),
                              stride=(first_conv_stride_y, first_conv_stride_x),
                              padding=(5, 3))

        # Relu layer
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout()

        # Reshape layer
        fc_in_features = 4000
        self.fc_reshape = Reshape([-1, fc_in_features])

        # Fully Connected layer
        self.fc = nn.Linear(in_features=fc_in_features, out_features=n_output)

        # Softmax
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Reshape input x
        x = self.conv_reshape(x)

        # Pass x through conv2d layer
        x_conv = self.conv(x)

        # Pass x through relu layer
        x_relu = self.relu(x_conv)

        # Pass x through dropout layer during tesing
        if self.training:
            x_dropout = self.dropout(x_relu)
        else:
            x_dropout = x_relu

        # Reshape x
        x_dropout = self.fc_reshape(x_dropout)

        # Pass x through fully connected layer
        x_fc = self.fc(x_dropout)

        # Pass x through softmax layer
        if self.training:
            y = F.log_softmax(x_fc,-1)
        else:
            y = self.softmax(x_fc)

        return y
    
class TinyConvPurged(nn.Module):
    def __init__(self, model_settings, n_input=1, n_output=4, num_filters=8):
        super(TinyConvPurged, self).__init__()
        self.first_filter_width = 8
        self.first_filter_height = 10
        self.first_filter_count = num_filters
        self.first_conv_stride_x = 2
        self.first_conv_stride_y = 2
        self.model_settings = model_settings

        self.input_frequency_size = self.model_settings['fingerprint_width']
        self.input_time_size = self.model_settings['spectrogram_length']
        self.W = self.input_frequency_size
        self.H = self.input_time_size
        self.C = 1

        # Calculate output dimensions after convolution
        self.out_h = ((self.H + 2 * 5 - self.first_filter_height) // self.first_conv_stride_y + 1)
        self.out_w = ((self.W + 2 * 3 - self.first_filter_width) // self.first_conv_stride_x + 1)
        
        # Reshape layer
        self.conv_reshape = Reshape([-1, self.C, self.H, self.W])

        # Conv2d layer
        self.conv = nn.Conv2d(in_channels=n_input,
                            out_channels=self.first_filter_count,
                            kernel_size=(self.first_filter_height, self.first_filter_width),
                            stride=(self.first_conv_stride_y, self.first_conv_stride_x),
                            padding=(5, 3))

        # Relu layer
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout()

        # Calculate fc input features based on conv output
        self.fc_in_features = self.first_filter_count * self.out_h * self.out_w
        self.fc_reshape = Reshape([-1, self.fc_in_features])

        # Fully Connected layer
        self.fc = nn.Linear(in_features=self.fc_in_features, out_features=n_output)

        # Softmax
        self.softmax = torch.nn.Softmax(dim=1)

    def get_fc_in_features(self):
        return self.first_filter_count * self.out_h * self.out_w

    def forward(self, x):
        # Reshape input x
        x = self.conv_reshape(x)

        # Pass x through conv2d layer
        x_conv = self.conv(x)

        # Pass x through relu layer
        x_relu = self.relu(x_conv)

        # Pass x through dropout layer during training
        if self.training:
            x_relu = self.dropout(x_relu)

        # Reshape x
        x_reshaped = self.fc_reshape(x_relu)

        # Pass x through fully connected layer
        x_fc = self.fc(x_reshaped)

        # Pass x through softmax layer
        if self.training:
            y = F.log_softmax(x_fc, -1)
        else:
            y = self.softmax(x_fc)

        return y

