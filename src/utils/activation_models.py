import torch
from torch import nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicCNN1(nn.Module):
    def __init__(self, input_channels, output_size):
        super(BasicCNN1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(8 * 7 * 7, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicCNN2(nn.Module):
    def __init__(self, input_channels, output_size):
        super(BasicCNN2, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BasicCNN2m(nn.Module):
    def __init__(self, input_channels, output_size):
        super(BasicCNN2m, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc2(F.relu(self.fc1(x)))
        return x

class MLP2bn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2bn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP2ln(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2ln, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x

class MLP2c(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, output_size):
        super(MLP2c, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size)
        self.fc2 = nn.Linear(input_size2, hidden_size)
        self.fc3 = nn.Linear(hidden_size*2, 32)
        self.fc_out = nn.Linear(32, output_size)

    def forward(self, input1, input2):
        x = F.relu(self.fc1(input1))
        y = F.relu(self.fc2(input2))
        z = F.relu(self.fc3(torch.cat((x, y), dim=1)))
        z = self.fc_out(z)
        return z

class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP2d(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, amount=0.1):
        super(MLP2d, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(amount)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(self.dropout(x))
        return x

class MLP3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(input_dim / num_heads)
        if self.attention_head_size == 0:
            self.num_heads = 1
            self.attention_head_size = input_dim
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(input_dim, self.all_head_size)
        self.key = nn.Linear(input_dim, self.all_head_size)
        self.value = nn.Linear(input_dim, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = mixed_query_layer.view(-1, self.num_heads, self.attention_head_size)
        key_layer = mixed_key_layer.view(-1, self.num_heads, self.attention_head_size)
        value_layer = mixed_value_layer.view(-1, self.num_heads, self.attention_head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.contiguous().view(-1, self.all_head_size)
        return context_layer


class TinyAttentionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads):
        super(TinyAttentionMLP, self).__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        attention_output = self.attention(x)
        x = x + attention_output  # Skip connection
        x = self.layer_norm1(x)

        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.layer_norm2(hidden)

        output = self.fc2(hidden)
        return output