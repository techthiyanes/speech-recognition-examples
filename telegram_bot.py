import os
import torch
import torchaudio
from pydub import AudioSegment
from telegram.ext import Updater, MessageHandler, Filters
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("anton-l/wav2vec2-large-xlsr-53-russian").to("cuda")
resampler = torchaudio.transforms.Resample(48_000, 16_000)


def voice_handler(update, context):
    src = str(update.message.voice.file_id) + '.ogg'
    dst = str(update.message.voice.file_id) + '.wav'
    context.bot.getFile(update.message.voice.file_id).download(src)
    AudioSegment.from_ogg(src).export(dst, format="wav")
    input_values = processor(resampler(torchaudio.load(dst)[0]).squeeze().numpy(), sampling_rate=16_000,
                             return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits
    os.remove(src)
    os.remove(dst)
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=processor.batch_decode(torch.argmax(logits, dim=-1))[0].lower())


updater = Updater(token='TOKEN')
updater.dispatcher.add_handler(MessageHandler(Filters.voice, voice_handler))
updater.start_polling()
