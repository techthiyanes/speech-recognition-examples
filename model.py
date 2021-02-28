import os
import torch
from ds_ctcdecoder import Alphabet, ctc_beam_search_decoder, Scorer
from torch import nn


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, n_feats):
        super(ResidualCNN, self).__init__()
        self.norm = nn.LayerNorm(n_feats)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2)

    def forward(self, x):
        x = self.norm(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return self.conv(self.dropout(self.gelu(x))) + x


class RNN(nn.Module):
    def __init__(self, rnn_dim, hidden_size, batch_first):
        super(RNN, self).__init__()
        self.norm = nn.LayerNorm(rnn_dim)
        self.gelu = nn.GELU()
        self.gru = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, num_layers=1,
                          batch_first=batch_first, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.gru(self.gelu(self.norm(x)))
        return self.dropout(x)


class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.res_cnn = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fc = nn.Linear(n_feats * 32, rnn_dim)
        self.rnn = nn.Sequential(*[
            RNN(rnn_dim=rnn_dim if i == 0 else rnn_dim * 2, hidden_size=rnn_dim, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.dense = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(rnn_dim, n_class)
        )
        self.alphabet = Alphabet(os.path.abspath("chars.txt"))
        self.scorer = Scorer(alphabet=self.alphabet,
                             scorer_path='librispeech.scorer', alpha=0.75, beta=1.85)

    def forward(self, x):
        x = self.res_cnn(self.cnn(x))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.permute(0, 2, 1)
        return self.dense(self.rnn(self.fc(x)))

    def beam_search_with_lm(self, xb):
        with torch.no_grad():
            out = self.forward(xb)
            softmax_out = out.softmax(2).cpu().numpy()
            char_list = []
            for i in range(softmax_out.shape[0]):
                char_list.append(ctc_beam_search_decoder(probs_seq=softmax_out[i, :], alphabet=self.alphabet,
                                                         scorer=self.scorer, beam_size=25)[0][1])
        return char_list
