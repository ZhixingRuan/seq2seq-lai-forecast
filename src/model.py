import torch
import torch.nn as nn

from log import get_logger

logger = get_logger(__name__)


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for encoder and decoder"""

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        # Add clamping to restrict their values to a reasonable range
        c_next = torch.clamp(c_next, min=-1e6, max=1e6)
        h_next = torch.clamp(h_next, min=-1e6, max=1e6)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device, init_type='zeros'):
        height, width = image_size
        if init_type == 'zeros':
            return (
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            )
        elif init_type == 'random':
            return (
                torch.rand(batch_size, self.hidden_dim, height, width, device=device)
                - 0.5,
                torch.rand(batch_size, self.hidden_dim, height, width, device=device)
                - 0.5,
            )
        else:
            raise ValueError('Unknown init_type')


class ConvLSTMSeq2Seq(nn.Module):
    """A basic Seq2Seq model with ConvLSTM cells"""

    def __init__(self, input_dim, target_dim, hidden_dim, kernel_size):
        super(ConvLSTMSeq2Seq, self).__init__()
        self.target_dim = target_dim
        self.encoder = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.decoder = ConvLSTMCell(target_dim, hidden_dim, kernel_size)
        self.fc = nn.Conv2d(hidden_dim, target_dim, kernel_size=1)

    def forward(self, input_seq, target_seq_length):
        batch_size, seq_len, _, height, width = input_seq.size()
        device = input_seq.device

        # Initialize hidden state and cell state
        h, c = self.encoder.init_hidden(batch_size, (height, width), device)

        # Encoder: process each time step
        for t in range(seq_len):
            h, c = self.encoder(input_seq[:, t, :, :, :], (h, c))

        outputs = []
        prev_output = torch.zeros(batch_size, self.target_dim, height, width, device=device)

        # Decoder: autoregressively generate the sequence
        for t in range(target_seq_length):
            h, c = self.decoder(prev_output, (h, c))
            out = self.fc(h)
            outputs.append(out)
            prev_output = out

        outputs = torch.stack(
            outputs, dim=1
        )  # (batch_size, target_seq_length, target_dim, height, width)
        return outputs


class ConvLSTMSeq2SeqDOY(nn.Module):

    def __init__(self, input_dim, target_dim, hidden_dim, kernel_size, dropout=None):
        super(ConvLSTMSeq2SeqDOY, self).__init__()
        # Updated input_dim for encoder to include DOY features
        self.target_dim = target_dim
        self.encoder = ConvLSTMCell(input_dim + 2, hidden_dim, kernel_size)
        self.decoder = ConvLSTMCell(target_dim, hidden_dim, kernel_size)
        self.fc = nn.Conv2d(hidden_dim + 2, target_dim, kernel_size=1)

    def forward(self, input_seq, doy_seq_x, doy_seq_y, target_seq_length):
        batch_size, seq_len, _, height, width = input_seq.size()
        device = input_seq.device

        sine_component = doy_seq_x[:, 0, :, :, :]
        cosine_component = doy_seq_x[:, 1, :, :, :]

        sine_component_y = doy_seq_y[:, 0, :, :, :]
        cosine_component_y = doy_seq_y[:, 1, :, :, :]

        # Initialize hidden state and cell state
        h, c = self.encoder.init_hidden(batch_size, (height, width), device)

        # Encoder:
        for t in range(seq_len):
            input_with_features = torch.cat(
                [input_seq[:, t, :, :, :],
                 sine_component[:, t, :, :].unsqueeze(1),
                 cosine_component[:, t, :, :].unsqueeze(1)],
                dim=1
            )
            h, c = self.encoder(input_with_features, (h, c))

        outputs = []
        prev_output = torch.zeros(batch_size, self.target_dim, height, width, device=device)

        # Decoder: autoregressively generate the sequence
        for t in range(target_seq_length):
            h, c = self.decoder(prev_output, (h, c))

            h_concat = torch.cat(
                [h,
                 sine_component_y[:, t, :, :].unsqueeze(1),
                 cosine_component_y[:, t, :, :].unsqueeze(1)],
                dim=1
            )

            out = self.fc(h_concat)
            outputs.append(out)
            prev_output = out

        outputs = torch.stack(outputs, dim=1)
        return outputs


class ConvLSTMSeq2SeqDOY2Stacked(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_dim, kernel_size, dropout=0.2):
        super().__init__()
        self.target_dim = target_dim
        self.dropout = nn.Dropout2d(p=dropout)

        # Encoder: 2 stacked cells
        self.encoder_1 = ConvLSTMCell(input_dim + 2, hidden_dim, kernel_size)
        self.encoder_2 = ConvLSTMCell(hidden_dim,    hidden_dim, kernel_size)

        # Decoder: 2 stacked cells
        self.decoder_1 = ConvLSTMCell(target_dim, hidden_dim, kernel_size)
        self.decoder_2 = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)

        self.fc = nn.Conv2d(hidden_dim + 2, target_dim, kernel_size=1)

    def forward(self, input_seq, doy_seq_x, doy_seq_y, target_seq_length):
        batch_size, seq_len, _, height, width = input_seq.size()
        device = input_seq.device

        sine_x   = doy_seq_x[:, 0]
        cosine_x = doy_seq_x[:, 1]
        sine_y   = doy_seq_y[:, 0]
        cosine_y = doy_seq_y[:, 1]

        # Initialize hidden/cell states for both layers
        h1, c1 = self.encoder_1.init_hidden(batch_size, (height, width), device)
        h2, c2 = self.encoder_2.init_hidden(batch_size, (height, width), device)

        # Encoder
        for t in range(seq_len):
            input_with_doy = torch.cat([
                input_seq[:, t],
                sine_x[:, t].unsqueeze(1),
                cosine_x[:, t].unsqueeze(1)
            ], dim=1)

            h1, c1 = self.encoder_1(input_with_doy, (h1, c1))  # layer 1
            h2, c2 = self.encoder_2(self.dropout(h1), (h2, c2)) # layer 2 with dropout

        # Decoder
        outputs = []
        prev_output = torch.zeros(batch_size, self.target_dim, height, width, device=device)

        for t in range(target_seq_length):
            h1, c1 = self.decoder_1(prev_output, (h1, c1))  # layer 1
            h2, c2 = self.decoder_2(self.dropout(h1), (h2, c2))

            h_concat = torch.cat([
                h2,                              # use TOP layer's hidden state
                sine_y[:, t].unsqueeze(1),
                cosine_y[:, t].unsqueeze(1)
            ], dim=1)

            out = self.fc(h_concat)
            outputs.append(out)
            prev_output = out

        return torch.stack(outputs, dim=1)