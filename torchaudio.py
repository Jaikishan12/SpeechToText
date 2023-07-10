import pyaudio
import wave
import torch
import zipfile
import torchaudio
from glob import glob

CHUNK=1024
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100

p=pyaudio.PyAudio()

stream=p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
print("SPEAK...")

frames=[]
try:
    while(True):
        data=stream.read(CHUNK)
        frames.append(data)

except KeyboardInterrupt:
    pass
print("Done record")
stream.stop_stream()
stream.close()
p.terminate()

wf=wave.open("output.wav",'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en', # also available 'de', 'es'
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

# download a single file, any format compatible with TorchAudio (soundfile backend)

test_files = glob('output.wav')
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

output = model(input)
for example in output:
    print(decoder(example.cpu()))