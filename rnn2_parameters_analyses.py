import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

PATH = "</path/to/>/nietzsche.txt"


def detach_from_history(h):
    if type(h) == torch.Tensor:
        return h.detach()

    return tuple(detach_from_history(v) for v in h)


class CharRnn(nn.Module):
    def __init__(self, vocab_size, n_fac, n_hidden, batch_size, layers=2, dropout=0.3):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.LSTM(n_fac, n_hidden, layers, dropout=dropout)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.n_hidden = n_hidden
        self.layers = layers
        self.init_hidden_state(batch_size)

    def init_hidden_state(self, batch_size):
        self.h = (
            torch.zeros(self.layers, batch_size, self.n_hidden).cuda(),
            torch.zeros(self.layers, batch_size, self.n_hidden).cuda(),
        )

    def forward(self, inp):
        inp = self.e(inp)
        b_size = inp[0].size(0)
        if self.h[0].size(1) != b_size:
            self.init_hidden_state(b_size)

        outp, h = self.rnn(inp, self.h)
        self.h = detach_from_history(h)

        return F.log_softmax(self.l_out(outp), dim=-1)


text = ""
with open(PATH, "r") as file:
    text = file.read().replace("\n", " ")

chars = sorted(list(set(text)))
int2char = dict(enumerate(chars))
char2int = {char: ind for ind, char in int2char.items()}

idx = [char2int[c] for c in text]

epochs = 60
seq_size = 42
hidden_size = 512
batch_size = 64

# PREPARE DATA
# non overlapping sets of characters, predict seq_size characters starting from position 1
in_text = np.array(
    [
        [idx[j + i] for i in range(seq_size)]
        for j in range(0, len(idx) - seq_size - 1, seq_size)
    ]
)
out_text = np.array(
    [
        [idx[j + i] for i in range(seq_size)]
        for j in range(1, len(idx) - seq_size - 1, seq_size)
    ]
)


def calculate_layers_losses():
    layers_losses = []
    for layers in range(1, 5):
        net = CharRnn(len(char2int), seq_size, hidden_size, batch_size, layers).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        for e in range(0, epochs):
            loss = 0
            net.init_hidden_state(batch_size)

            for b in range(0, in_text.shape[0] // batch_size):
                idxs = (
                    torch.LongTensor(in_text[b * batch_size : (b + 1) * batch_size, :seq_size])
                    .transpose(0, 1)
                    .cuda()
                )
                lbls = (
                    torch.LongTensor(out_text[b * batch_size : (b + 1) * batch_size])
                    .squeeze()
                    .cuda()
                )

                res = net(idxs)
                loss = criterion(res.permute(1, 2, 0), lbls)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print("Epoch {}, loss {}".format(e + 1, loss.item()))

        layers_losses.append(loss.item())

    return layers_losses


def calculate_other_losses():
    other_losses = dict()
    for h_size in range(256, 1152, 128):
        other_losses[h_size] = []
        for d_out in np.arange(0.1, 1, 0.1):
            net = CharRnn(len(char2int), seq_size, h_size, batch_size, 2, d_out).cuda()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

            for e in range(0, epochs):
                loss = 0
                net.init_hidden_state(batch_size)

                for b in range(0, in_text.shape[0] // batch_size):
                    input_idxs = (
                        torch.LongTensor(in_text[b * batch_size : (b + 1) * batch_size, :seq_size])
                            .transpose(0, 1)
                            .cuda()
                    )
                    target_idxs = (
                        torch.LongTensor(out_text[b * batch_size : (b + 1) * batch_size])
                            .squeeze()
                            .cuda()
                    )

                    res = net(input_idxs)
                    loss = criterion(res.permute(1, 2, 0), target_idxs)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                print("Epoch {}, loss {}, hidden size {}, dropout {}".format(e + 1, loss.item(), h_size, d_out))

            other_losses[h_size].append(loss.item())

    return other_losses


layers_losses = calculate_layers_losses()
print("Layers losses: ", layers_losses)

other_losses = calculate_other_losses()
print("Dropout and hidden layer size losses: ", other_losses)

# PLOT
fig = plt.figure()
ax = plt.axes(projection="3d")

hidden_size_as_x = []
dropout_as_y = []
loss_as_z = []

for h_size, d_outs in other_losses.items():
    for i in range(len(d_outs)):
        hidden_size_as_x.append(h_size)
        dropout_as_y.append(0.1 * (i + 1))
        loss_as_z.append(d_outs[i])

ax.scatter(hidden_size_as_x, dropout_as_y, loss_as_z)
plt.show()
