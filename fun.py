from trainer import *
import wget
# wget.download("https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cHUnjMjsH5s4VZCIcO5u4HMRLVyfe7gm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cHUnjMjsH5s4VZCIcO5u4HMRLVyfe7gm" -O UrduWords.zip && rm -rf /tmp/cookies.txt)
# wget.download('https://raw.githubusercontent.com/ZdsAlpha/Z/main/trainer.py')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torchvision
import torchvision.models as models
import torchaudio

torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
from matplotlib import pyplot as plt
from zipfile import ZipFile
import numpy as np
import math
import random
import copy
import glob
import itertools
import jellyfish
import os
path = os.getcwd()
print(path)
# def extract_dataset(file_name):
#     with ZipFile(file_name, 'r') as zip:
#     # printing all the contents of the zip file
#         zip.printdir()
#         # extracting all the files
#         print('Extracting all the files now...')
#         zip.extractall()
#         print('Done!')
        
#extract_dataset('ar.zip')
def character_to_labels(audio_files):
    #labels = [os.path.split("/")[1].replace(" ", "") for path in audio_files]
    labels = os.listdir(path + '\\ar')
    chars = []
    for label in labels:
        for c in label:
            if not c in chars:
                chars.append(c)
                chars = sorted(chars)
                # print(c)
    return chars,labels
zero = []
dataset = "ar"
audio_files = glob.glob(dataset + "/*/*.wav")
# print("length of audio files  =  "+str(len(audio_files)))
chars,labels = character_to_labels(audio_files)
# print("Total audio files = " + str(len(audio_files)))
# print("Total characters = " + str(len(chars)))
# print(labels)
def extract_waveforms(audio_files):
    max_length = 16384 # length greater  than max length
    maxx = 0
    waves = []
    for audio in audio_files:
        waveform = torchaudio.load(audio)
        
        #waveform = torch.stack(asr).numpy()
        plt.figure()
        plt.plot(waveform.t().numpy())
        if (waveform.shape[1] > maxx):
            maxx = waveform.shape[1]
        padded_waveform = torch.zeros((1, max_length))
        selected_length = min(max_length, waveform.shape[1])
        padded_waveform[:, :selected_length] = waveform[:, :selected_length]
        waves.append(padded_waveform)
        
    print("max  length of audio file = " + str(maxx))
    return waves
waves = extract_waveforms(audio_files)
plt.figure()
plt.plot(waves[2499].t().numpy())
def extract_mfcc_features(waves):
    # Extracting MFCC features
    x = []
    mel_transform = torchaudio.transforms.MFCC()
    for wave in waves:
        spectrum = mel_transform(wave).squeeze()
        x.append(spectrum)
    x = torch.stack(x)
    return x
x = extract_mfcc_features(waves)
plt.figure()
plt.imshow(x[0].numpy())
max_word = 10

def encode_label(label):
  label = [chars.index(c) + 1 for c in label]
  label = label + [0] * (max_word - len(label))
  return label

def decode_label(label):
  label = [k for k, g in itertools.groupby(list(label))]
  output = ""
  for i in label:
    if i > 0 and i <= len(chars):
      output += chars[i - 1]
  return output

def covert_label_to_indices(labels):
    # Converting all labels to indices
    y = []
    y_len = []
    for label in labels:
        y_len.append(len(label))
        label = encode_label(label)
        y.append(torch.LongTensor(label))
    y = torch.stack(y)
    y_len = torch.LongTensor(y_len)
    return y,y_len
y,y_len = covert_label_to_indices(labels)
dataset = TensorDataset(x, y, y_len)
train_set, test_set = split_dataset(dataset, ratio=0.9, shuffle=True)
# Defining model
output_length = 37
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv1d(40, 64, 5),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Conv1d(64, 64, 5),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.MaxPool1d(2, 2),
        nn.Dropout(0.5),
    ) # (*, 64, (82 - 4 - 4) / 2)
    self.rnn = nn.GRU(64, 64, 2, batch_first=True, bidirectional=True, dropout=0.5) # (*, 128, 37)
    self.classify = nn.Linear(128, len(chars) + 1)

  def forward(self, x):
    x = self.conv(x).permute(0, 2, 1)
# Initialzing model
if torch.cuda.is_available(): 
  device = torch.device("cuda") 
  print("working on gpu")
  model = Model().to(device)
else:
  device = torch.device("cpu") 
  print("working on cpu")
  model = Model().to(device)
# Defining loss function
def loss_func(model, batch, scope):
  x, y, y_len = batch
  scores = model(x)
  scores = scores.permute(1, 0, 2)
  x_len = torch.LongTensor([output_length] * len(x))
  loss = F.ctc_loss(scores, y, x_len, y_len)
  return loss, scores
# Print predicted squence after every 50 epochs
def on_batch(scope):
  if scope["iteration"] % 50 == 0:
    sequence = scope["output"][:, 0, :].detach().cpu().numpy()
    sequence = decode_label(sequence.argmax(1))
    print(sequence)
    print("Current Loss = " + str(float(scope["loss"])))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# Training model
train(model, loss_func, train_set, test_set, optimizer, device=0, batch_size=16,on_train_batch=on_batch, epochs=200)
# Evaluating on test-set
model.eval()
# Evaluating on test-set
acc = []
model.eval()
# Install jellyfish
def crr(s1, s2):
  return 1 - jellyfish.levenshtein_distance(s1, s2) / len(s1)
for x, y, _ in test_set:
  with torch.no_grad():
    gt = decode_label(y)
    pred = decode_label(model(x.unsqueeze(0).to(0)).cpu()[0].argmax(1))
    sim = crr(gt, pred)
    acc.append(sim)
    print("Predicted = " + pred)
    print("Ground Truth = " + gt)
    print("CRR = " + str(sim))
  print("\n")
print("Averag CRR = " + str(sum(acc) / len(acc)))
