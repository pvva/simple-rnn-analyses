import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


PATH = "</path/to/>/nietzsche.txt"


def detach_from_history(h):
    if type(h) == torch.Tensor:
        return h.detach()

    return tuple(detach_from_history(v) for v in h)


def generateNextChar(charNet, phraze):
    idxs = np.empty((1, seq_size))
    idxs[0] = np.array([char2int[c] for c in phraze])

    res = charNet(torch.LongTensor(idxs).transpose(0, 1).cuda())
    _, t_idxs = torch.max(res, dim=2)

    # take last char of sequence
    return int2char[t_idxs.detach().cpu().numpy()[-1][0]]


def generateText(charNet, phraze, numChars):
    cText = phraze
    for i in range(0, numChars):
        cText += generateNextChar(charNet, cText[i:])

    return cText


class CharRnn(nn.Module):
    def __init__(self, vocab_size, n_fac, n_hidden, batch_size, layers=2):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.LSTM(n_fac, n_hidden, layers, dropout=0.1)
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

net = CharRnn(len(char2int), seq_size, hidden_size, batch_size).cuda()
lr = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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

print(in_text.shape)
print(out_text.shape)

# TRAIN
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
        # permute axes for cross entropy calculation, as it expects embedding size at index 1
        # embedding size is the same as vocabulary size (vocab_size)
        # (seq_size, batch_size, embedding_size) => (batch_size, embedding_size, seq_size)
        loss = criterion(res.permute(1, 2, 0), target_idxs)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    print("Epoch {}, loss {}".format(e + 1, loss.item()))

# GENERATE
print(text[:seq_size], "=>", generateText(net, text[:seq_size], 1500))
