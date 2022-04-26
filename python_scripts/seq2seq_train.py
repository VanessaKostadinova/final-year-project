import json
import os
import gensim
import matplotlib.pyplot as plt
from gensim import downloader
import torch
import torch.optim
import torch.nn as nn
import re
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from seq2seq_model import Encoder, AttnDecoder, Seq2Seq

class BVHDataset(Dataset):
    def __init__(self, list_of_names):
        self.transcripts = []
        self.bvh_files = []

        bvh_path = "F:\\fyp\\cut_data_openpose\\"
        transcript_path = "F:\\fyp\\transcript_aligned\\"

        for name in list_of_names:
            if os.path.exists(transcript_path + name + ".json") & os.path.exists(
                    bvh_path + name + "ML_BVH.txt"):
                with open(transcript_path + name + ".json", "r") as json_file:
                    transcript_json = json.load(json_file)
                    if "words" in transcript_json.keys():
                        transcript_words = [normalise_string(e["word"]) for e in transcript_json["words"]]
                        self.transcripts.append(transcript_words)
                    else:
                        continue

                with open(bvh_path + name + "ML_BVH.txt", "r") as bvh_file:
                    bvh_rows = []
                    for line in bvh_file:
                        string_contents = line.split(",")
                        float_contents = []
                        for x in string_contents:
                            float_contents.append(float(x))
                        bvh_rows.append(float_contents)
                    if len(bvh_rows) == 0:
                        self.transcripts.remove(transcript_json)
                    else:
                        self.bvh_files.append(bvh_rows)

    def __len__(self):
        return len(self.bvh_files)

    def __getitem__(self, index):
        return [self.transcripts[index], self.bvh_files[index]]

def normalise_string(string):
    string = string.lower()
    string = re.sub(r"[\d+]", "", string)
    string = re.sub(r"[^\w\s]", "", string)
    string = string.strip()
    return string

def custom_loss(output, target):
    n_element = output.numel()

    # MSE
    l1_loss = F.l1_loss(output, target)
    l1_loss *= 5

    # continuous motion
    diff = [abs(output[:, n] - output[:, n - 1]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    cont_loss *= 0.1

    # motion variance
    norm = torch.norm(output, 2, 1)
    var_loss = -torch.sum(norm) / n_element
    var_loss *= 0.5

    l = l1_loss + cont_loss + var_loss

    # inspect loss terms
    global loss_i
    if loss_i == 100:
        loss_i = 0
    loss_i += 1

    return l

num_epochs = 19
batch_size = 1

load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = gensim.downloader.load('glove-wiki-gigaword-300')

num_joints = 22
hidden_size = 512
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

data = BVHDataset(os.listdir("F:\\fyp\\cut_data_openpose\\"))

train_iterator = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, pin_memory=True)

encoder = Encoder(embedding_model, hidden_size, num_layers, encoder_dropout, device).to(device)
decoder = AttnDecoder(num_joints, hidden_size, num_layers, decoder_dropout, device).to(device)

seq2seq = Seq2Seq(encoder, decoder, hidden_size, device).to(device).float()

loss_fn = nn.MSELoss().to(device).float()

optimiser = optim.Adam(seq2seq.parameters(), lr=0.001, betas=(0.5, 0.8))

loss_i = 0

if load_model:
    seq2seq.load_state_dict(torch.load("./model_gru_arm1.pt"))
    seq2seq.train()

epoch_losses = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs}')
    epoch_loss = 0

    for i, (inputs, targets) in enumerate(train_iterator):
        optimiser.zero_grad()
        yhat = seq2seq.forward(inputs, targets, num_joints)
        with torch.cuda.amp.autocast():
            loss = custom_loss(yhat, torch.tensor(targets, dtype=torch.float, device=device))
        scaler.scale(loss).backward()
        scaler.step(optimiser)

        scaler.update()
        epoch_loss += loss.item()

    torch.save(seq2seq.state_dict(), "./model_gru_arm1.pt")
    print(epoch_loss / len(train_iterator))
    epoch_losses.append(epoch_loss / len(train_iterator))

print(epoch_losses)
plt.plot(epoch_losses)
plt.show()