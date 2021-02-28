import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
import torch
import torchaudio
from jiwer import wer
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import deepspeed

config = {
    "train_batch_size": 8,
    "fp16": {
        "enabled": True,
        "min_loss_scale": 1,
        "opt_level": "O3"
    },
    "zero_optimization": {
        "stage": 2,
        "cpu_offload": True,
        "cpu_offload_params": True,
        "contiguous_gradients": True,
        "overlap_comm": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-6
        }
    }
}
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model, optimizer, _, _ = deepspeed.initialize(config_params=config, model=model, model_parameters=model.parameters())
test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)

decoded = []
original = []

for waveform, _, utterance, _, _, _ in test_dataset:
    input_values = tokenizer(waveform.numpy().squeeze(0), return_tensors="pt", padding="longest").input_values
    decoded.append(tokenizer.batch_decode(torch.argmax(model(input_values.to(torch.float16).to("cuda")).logits, dim=-1))[0])
    original.append(utterance)

# 2.7712264150943398
print(wer(original, decoded) * 100)
