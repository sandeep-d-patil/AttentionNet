import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class downfirst(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downfirst, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.Conv2d(in_ch, in_ch, kernel_size=(2,2), stride=(2,2)),
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, (2, 2), stride=(2, 2))

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: str):
        super(TemporalAttention, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.hidden_attn = nn.Parameter(torch.randn(1, 128))
        self.input_attn = nn.Parameter(torch.randn(1, 128))
        self.attn_combine = nn.Parameter(torch.randn(1, 128))
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=False,
            num_layers=1,
        )

    def forward(self, input_tensor):

        hidden_tensor = self.initHidden(batch_size=input_tensor.size(1))
        input_tensor = input_tensor.permute(1, 0, 2)
        # global attn_output, attn_combine_fc
        attention_output = torch.zeros(
            1, input_tensor.size(0), input_tensor.size(2)
        ).to(device=self.device)
        seq_len = input_tensor.size(1)
        for t in range(seq_len):
            cur_inp_layer = input_tensor[:, t, :]
            input_fc = torch.mul(cur_inp_layer, self.input_attn)
            # hidden_tensor[0] gives the hidden state of the lstm
            hidden_fc = torch.mul(hidden_tensor[0], self.hidden_attn)
            attn_sum = torch.add(input_fc, hidden_fc)
            attn_combine_fc = torch.sigmoid(torch.mul(attn_sum, self.attn_combine))
            cur_inp_layer_res = cur_inp_layer.unsqueeze(0)
            attention_t = torch.mul(attn_combine_fc, cur_inp_layer_res)
            attention_output += attention_t

        output_tensor, hidden_tensor = self.lstm(attention_output, hidden_tensor)

        return output_tensor, hidden_tensor

    def initHidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size).cuda(),
            torch.zeros(1, batch_size, self.hidden_size).cuda(),
        )


# class TemporalAttention(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, device: str="cpu"):
#         super(TemporalAttention, self).__init__()

#         self.device = device
#         self.hidden_size = hidden_size
#         self.input_size = input_size

#         self.trainable_constant_h = nn.Parameter(torch.randn(1)).to(self.device)
#         self.trainable_constant_i = nn.Parameter(torch.randn(1)).to(self.device)
#         self.trainable_constant_a = nn.Parameter(torch.randn(1)).to(self.device)
#         self.lstm = nn.LSTM(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size,
#             batch_first=False,
#             num_layers=1,
#         )

#     def forward(self, input_tensor: torch.Tensor):

#         attention_output = torch.zeros(1, input_tensor.size(0), input_tensor.size(2)).to(device=self.device)
#         lstm_hidden = self.initHidden(batch_size=input_tensor.size(0))
#         input_tensor = input_tensor.permute(1, 0, 2)
#         seq_len = input_tensor.size(0)

#         for t in range(seq_len):
#             input_t = input_tensor[:, t, :]
#             input_o = torch.mul(input_t, self.trainable_constant_i)
#             hidden_o = torch.mul(lstm_hidden[0], self.trainable_constant_h)
#             intermediate_sum = torch.add(input_o, hidden_o)
#             attention_o = torch.sigmoid(
#                 torch.mul(intermediate_sum, self.trainable_constant_a)
#             )
#             input_t_flatten = input_t.unsqueeze(0)
#             attention_t = torch.mul(attention_o, input_t_flatten)
#             attention_output += attention_t
#             print("attention_output shape", attention_output.shape)

#         lstm_output, lstm_hidden = self.lstm(attention_output, lstm_hidden)

#         return lstm_output, lstm_hidden

# # TODO convert cuda to initialized device type
# def initHidden(self, batch_size):
#     return (
#         torch.zeros(1, batch_size, self.hidden_size).to(device=self.device),
#         torch.zeros(1, batch_size, self.hidden_size).to(device=self.device),
#     )


class SpatioTemporalAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: str = "cpu"):
        super(SpatioTemporalAttention, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

        self.trainable_linear_h = nn.Parameter(torch.randn(1, 128))
        self.trainable_linear_i = nn.Parameter(torch.randn(1, 128))
        self.trainable_linear_a = nn.Parameter(torch.randn(1, 128))
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=False,
            num_layers=1,
        )

    def forward(self, input_tensor: torch.Tensor):

        attention_output = torch.zeros(1, input_tensor.size(0), input_tensor.size(2))
        hidden_tensor = self.initHidden(batch_size=input_tensor.size(0))
        input_tensor = input_tensor.permute(1, 0, 2)
        input_length = input_tensor.size(0)

        for t in range(input_length):
            input_t = input_tensor[t, :, :]
            input_o = torch.mul(input_t, self.trainable_linear_i)
            hidden_o = torch.mul(hidden_tensor[0], self.trainable_linear_h)
            intermediate_sum = torch.add(input_o, hidden_o)
            attention_o = torch.sigmoid(
                torch.mul(intermediate_sum, self.trainable_linear_a)
            )
            input_t_flatten = input_t.unsqueeze(0)
            attention_t = torch.mul(attention_o, input_t_flatten)
            attention_output += attention_t

        lstm_output, lstm_hidden = self.lstm(attention_output, hidden_tensor)

        return lstm_output, lstm_hidden

    def initHidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size).to(device=self.device),
            torch.zeros(1, batch_size, self.hidden_size).to(device=self.device),
        )


class SpatioTemporalFCAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: str = "cpu"):
        super(SpatioTemporalFCAttention, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

        self.trainable_fc_h = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )
        self.trainable_fc_i = nn.Linear(
            in_features=self.input_size, out_features=self.input_size
        )
        self.trainable_fc_a = nn.Linear(
            in_features=self.input_size, out_features=self.input_size
        )
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=False,
            num_layers=1,
        )

    def forward(self, input_tensor: torch.Tensor):

        attention_output = torch.zeros(1, input_tensor.size(0), input_tensor.size(2))
        hidden_tensor = self.initHidden(batch_size=input_tensor.size(0))
        input_tensor = input_tensor.permute(1, 0, 2)
        input_length = input_tensor.size(0)

        for t in range(input_length):
            input_t = input_tensor[t, :, :]
            input_o = self.trainable_fc_i(input_t)

            hidden_o = self.trainable_fc_h(hidden_tensor[0])

            intermediate_sum = torch.add(input_o, hidden_o)
            attention_o = torch.sigmoid(self.trainable_fc_a(intermediate_sum))
            input_t_flatten = input_t.unsqueeze(0)
            attention_t = torch.mul(attention_o, input_t_flatten)
            attention_output += attention_t

        lstm_output, lstm_hidden = self.lstm(attention_output, hidden_tensor)

        return lstm_output, lstm_hidden

    # TODO convert cuda to initialized device type
    def initHidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size).to(device=self.device),
            torch.zeros(1, batch_size, self.hidden_size).to(device=self.device),
        )


if __name__ == "__main__":

    # Output from Conv2d down4 layer is (time_step, batch_size, channels, height, width)
    input_tensor = torch.randn(5, 10, 512, 8, 16)
    conv_layer = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(1, 1))
    x = torch.unbind(input_tensor, dim=1)

    conv_output = []
    for item in x:

        x1 = conv_layer(item)
        conv_output.append(x1.unsqueeze(0))
    conv_output = torch.cat(conv_output, dim=0)
    print(conv_output.shape)
    attention_module = TemporalAttention(input_size=128, hidden_size=128, device="cpu")
    input_tensor = torch.flatten(conv_output, start_dim=2)
    output_tensor, hidden_tensor = attention_module(input_tensor)
    print(output_tensor.shape)
    output_tensor = output_tensor.permute(1, 0, 2)
    output_tensor_new = output_tensor.reshape(
        output_tensor.shape[0], output_tensor.shape[1], 8, 16
    )
    print(output_tensor_new.shape)
    outconv = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(1, 1))
    output = outconv(output_tensor_new)
    print(output.shape)
