import os
import sys

import Levenshtein as leven
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torchaudio
from colorama import Fore
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
from model import SpeechRecognitionModel
from jiwer import wer

dev = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================= DATASET PARAMETERS =====================================================
target_dir = "./data"
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
train_dataset = torchaudio.datasets.LIBRISPEECH(target_dir, url="train-clean-100", download=True)
test_dataset = torchaudio.datasets.LIBRISPEECH(target_dir, url="test-clean", download=True)
classes = "' abcdefghijklmnopqrstuvwxyz"

# ============================================= TRANSFORMATIONS ========================================================
audio_transforms = torchaudio.transforms.MelSpectrogram()
num_to_char_map = {c: i for i, c in enumerate(list(classes))}
char_to_num_map = {v: k for k, v in num_to_char_map.items()}
str_to_num = lambda text: [num_to_char_map[c] for c in text]
num_to_str = lambda labels: ''.join([char_to_num_map[i] for i in labels])

# ============================================= ALPHABET FILE ==========================================================
text_file = open("chars.txt", "w", encoding='utf-8')
text_file.write('\n'.join(list(classes)))
text_file.close()


# ============================================= COLLATE FUNCTION =======================================================
def collate(data):
    spectrograms = [audio_transforms(waveform).squeeze(0).permute(1, 0) for (waveform, _, utterance, _, _, _) in data]
    labels = [torch.Tensor(str_to_num(utterance.lower())) for (waveform, _, utterance, _, _, _) in data]
    input_lengths = [spec.shape[0] // 2 for spec in spectrograms]
    label_lengths = [len(label) for label in labels]
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).permute(0, 1, 3, 2)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths


# ============================================= PREPARING DATASET ======================================================
train_batch_size = 40
validation_batch_size = 40
torch.manual_seed(7)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
validation_loader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate, pin_memory=True)
# ================================================= MODEL ==============================================================
model = SpeechRecognitionModel(n_cnn_layers=7, n_rnn_layers=5, rnn_dim=512, n_class=len(classes) + 1, n_feats=128).to(dev)


# ================================================ TRAINING MODEL ======================================================
def fit(model, epochs, train_data_loader, valid_data_loader):
    best_leven = 1000
    optimizer = optim.AdamW(model.parameters(), 5e-4)
    len_train = len(train_data_loader)
    loss_func = nn.CTCLoss(blank=len(classes)).to(dev)
    for i in range(1, epochs + 1):
        # ============================================ TRAINING ========================================================
        batch_n = 1
        train_levenshtein = 0
        len_levenshtein = 0
        all_train_decoded = []
        all_train_actual = []
        for spectrograms, labels, input_lengths, label_lengths in tqdm(train_data_loader,
                                                                       position=0, leave=True,
                                                                       file=sys.stdout,
                                                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                                                                               Fore.GREEN, Fore.RESET)):
            model.train()
            spectrograms, labels = spectrograms.to(dev), labels.to(dev)
            optimizer.zero_grad()
            loss_func(model(spectrograms).log_softmax(2).permute(1, 0, 2), labels, input_lengths, label_lengths).backward()
            optimizer.step()
            # ================================== TRAINING LEVENSHTEIN DISTANCE =========================================
            if batch_n > (len_train - 5):
                model.eval()
                with torch.no_grad():
                    decoded = model.beam_search_with_lm(spectrograms)
                    for j in range(0, len(decoded)):
                        actual = num_to_str(labels.cpu().numpy()[j][:label_lengths[j]].tolist())
                        all_train_decoded.append(decoded[j])
                        all_train_actual.append(actual)
                        train_levenshtein += leven.distance(decoded[j], actual)
                        len_levenshtein += label_lengths[j]

            batch_n += 1
        # ============================================ VALIDATION ======================================================
        model.eval()
        with torch.no_grad():
            val_levenshtein = 0
            target_lengths = 0
            all_valid_decoded = []
            all_valid_actual = []
            for spectrograms, labels, input_lengths, label_lengths in tqdm(valid_data_loader,
                                                                           position=0, leave=True,
                                                                           file=sys.stdout,
                                                                           bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                                                                                   Fore.BLUE, Fore.RESET)):
                spectrograms, labels = spectrograms.to(dev), labels.to(dev)
                decoded = model.beam_search_with_lm(spectrograms)
                for j in range(0, len(decoded)):
                    actual = num_to_str(labels.cpu().numpy()[j][:label_lengths[j]].tolist())
                    all_valid_decoded.append(decoded[j])
                    all_valid_actual.append(actual)
                    val_levenshtein += leven.distance(decoded[j], actual)
                    target_lengths += label_lengths[j]

        print('Epoch {}: Levenshtein Train: {:.4f} Valid: {:.4f} | WER  Train: {:.4f} | Valid: {:.4f}'
              .format(i, train_levenshtein / len_levenshtein, val_levenshtein / target_lengths,
                      wer(all_train_actual, all_train_decoded), wer(all_valid_actual, all_valid_decoded)), end='\n')
        # ============================================ SAVE MODEL ======================================================
        if (val_levenshtein / target_lengths) < best_leven:
            torch.save(model.state_dict(), f=str((val_levenshtein / target_lengths) * 100).replace('.', '_') + '_' + 'model.pth')
            best_leven = val_levenshtein / target_lengths


summary(model, (1, 128, 1344))
print(model)
print("Training...")
# model.load_state_dict(torch.load('./weights.pth'))
fit(model=model, epochs=25, train_data_loader=train_loader, valid_data_loader=validation_loader)


# ============================================ TESTING =================================================================
def batch_predict(model, valid_dl, up_to):
    model.eval()
    spectrograms, labels, input_lengths, label_lengths = iter(valid_dl).next()
    with torch.no_grad():
        outs = model.beam_search_with_lm(spectrograms.to(dev))
        for i in range(len(outs)):
            actual = num_to_str(labels.numpy()[i][:label_lengths[i]].tolist())
            predicted = outs[i]
            # ============================================ SHOW IMAGE ==================================================
            img = spectrograms.log2()[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            f, ax = plt.subplots(1, 1)
            mpl.rcParams["font.size"] = 8
            ax.imshow(img)
            mpl.rcParams["font.size"] = 10
            plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(actual))
            plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(predicted))
            f.set_size_inches(18, 4)
            print('actual: {}'.format(actual))
            print('predicted:   {}'.format(predicted))
            if i + 1 == up_to:
                break
    plt.show()


batch_predict(model=model, valid_dl=validation_loader, up_to=20)
