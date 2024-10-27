import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pandas as pd
import os
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import csv
# import datasets


# from torchtext.data import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

# pytorch loads the IMBD data for you, but if you want to look at the raw format, you can download from here:
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True



def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}

def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    length = len(tokens)
    return {"tokens": tokens, "length": length}

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)



def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)



def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


class Dubin_Path_RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        trajectory_length
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.lstm = nn.LSTM(
            output_dim,
            hidden_dim, # hidden_dim
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True
        )
        # self.lstm = nn.LSTM(
        #     input_dim,
        #     hidden_dim, # hidden_dim
        #     n_layers,
        #     bidirectional=bidirectional,
        #     dropout=dropout_rate,
        #     batch_first=True
        # )

        # self.output_layer = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.output_layer = nn.Sequential(
             nn.Linear(hidden_dim, 256),
             nn.ReLU(),
             nn.Linear(256, output_dim)
        )

        # self.output_layer = nn.Linear(hidden_dim * 2 if bidirectional else output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, initial_conditions, ground_truth, teacher_forcing_prob, hidden_in):
       
       batch_size = initial_conditions.size(0)
       lstm_input = self.fc(initial_conditions).unsqueeze(1)
    #    cell = torch.zeros_like(hidden)

       outputs = []
    #    lstm_input = torch.zeros(batch_size, 1, hidden.size(-1))
    #    lstm_input = hidden

       for i in range(ground_truth.size(1)):
           out, hidden_in = self.lstm(lstm_input, hidden_in)
           trajectory_point = self.output_layer(out.squeeze(1))
           outputs.append(trajectory_point.unsqueeze(1))

           use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
           if use_teacher_forcing and ground_truth is not None:
                lstm_input = ground_truth[:, i].unsqueeze(1)
           else:
                lstm_input = trajectory_point.unsqueeze(1)
        #    lstm_input = out
       return torch.cat(outputs, dim=1)
    



        # embedded = self.dropout(self.embedding(ids))
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(
        #     embedded, length, batch_first=True, enforce_sorted=False
        # ) # makes padding ignored by RNN
        # packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # # hidden = [n layers * n directions, batch size, hidden dim]
        # # cell = [n layers * n directions, batch size, hidden dim]
        # output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # # output = [batch size, seq len, hidden dim * n directions]
        # if self.lstm.bidirectional:
        #     hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        #     # hidden = [batch size, hidden dim * 2]
        # else:
        #     hidden = self.dropout(hidden[-1])
        #     # hidden = [batch size, hidden dim]
        # prediction = self.fc(hidden)
        # # prediction = [batch size, output dim]
        # return prediction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)

class TrajectoryDataset(Dataset):
    def __init__(self, initial_conditions, trajectories, train_params_mean, train_params_std, train_paths_mean, train_paths_std):
        self.initial_conditions = initial_conditions  # Shape: (num_samples, num_params)
        self.trajectories = trajectories  # Shape: (num_samples, 1000, 3)
        self.params_mean = train_params_mean
        self.params_std = train_params_std
        self.traj_mean = train_paths_mean
        self.traj_std = train_paths_std

    def __len__(self):
        return len(self.initial_conditions)

    def __getitem__(self, idx):
        params = (self.initial_conditions[idx] - self.params_mean) / self.params_std
        trajectory = (self.trajectories[idx] - self.traj_mean) / self.traj_std
        return params, trajectory #self.trajectories[idx]

if __name__ == '__main__':

    max_length = 175 #256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Will
    # parameters_path = "C:/Users/willi/Documents/GitHub/ML_for_Robots_hw3/data/parameters.xls"
    # psi_end_path = "C:/Users/willi/Documents/GitHub/ML_for_Robots_hw3/data/psi_end.xls"
    # data_path = "C:/Users/willi/Documents/GitHub/ML_for_Robots_hw3/data/path.mat"
    # file_name = "dubin_path"

    # Chase
    parameters_path = "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/parameters.xls"
    psi_end_path = "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/psi_end.xls"
    data_path = "c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/data/path.mat"
    file_name = "dubin_path"

    df_params = pd.read_excel(parameters_path)
    df_psi = pd.read_excel(psi_end_path)

    # Convert from pandas dataframe to numpy array
    parameters_data = df_params.values
    psi_end_data = df_psi.values

    # remove initial position, steplength, and r_min from data
    parameters_data = parameters_data[:, 3:7]


    # temp_params = torch.tensor([], dtype=torch.float32).to(device)
    # temp_psi_end = torch.tensor([], dtype=torch.float32).to(device)

    params_counter = 0
    psi_end_counter = 0

    for list in parameters_data: #range(0, parameters_data.shape[0] + 1):
        if params_counter == 0:
            temp_params = torch.tensor(list, dtype=torch.float32).to(device)
        else:
            temp_params = torch.vstack((temp_params, torch.tensor(list, dtype=torch.float32).to(device)))
        params_counter += 1

    parameters_data = temp_params

    for list in psi_end_data: #range(0, parameters_data.shape[0] + 1):
        if psi_end_counter == 0:
            temp_psi_end = torch.tensor(list, dtype=torch.float32).to(device)
        else:
            temp_psi_end = torch.vstack((temp_psi_end, torch.tensor(list, dtype=torch.float32).to(device)))
        psi_end_counter += 1

    psi_end_data = temp_psi_end
    
    

    # List to store the data from all files
    # dataframes = []
    # paths = []
    # paths = torch.tensor(paths, dtype=torch.float32).to(device)

    # paths = np.zeros((10000, 1000, 3)) # depth, rows, columns
    paths = np.zeros((10000, 175, 3))

    df = sio.loadmat(data_path)
    df = df['master_path']
    # Loop through all files in the folder
    for i in range(df.shape[2]):
        if i == 0:
            # paths = torch.tensor(df[:, :, i], dtype=torch.float32).to(device)
            paths[i, :, :] = df[0:175, :, i]
        else:
            # paths = torch.vstack((paths, torch.tensor(df[:,:,i], dtype=torch.float32).to(device)))
            # paths[:, :, i] = torch.tensor(df[:, :, i], dtype = torch.float32).to(device)
            paths[i, :, :] = df[0:175, :, i]
    paths = torch.tensor(paths, dtype=torch.float32).to(device)
    # paths = torch.stack(paths, dim=2)
    
    train_indices = int(((i + 1) * 0.8) - 1)
    # val_indices = int(((i + 1) * 0.2) - 1)

    train_paths = paths[0:train_indices, :, :]
    val_paths = paths[train_indices + 1:-1, :, :]

    train_params = parameters_data[0:train_indices]
    val_params = parameters_data[train_indices + 1:]

    train_psi_end = psi_end_data[0:train_indices]
    val_psi_end = psi_end_data[train_indices + 1:]

    train_params_mean = train_params.mean(dim=0)
    train_params_std = train_params.std(dim=0)

    train_paths_mean = train_paths.mean(dim=(0,1))
    train_paths_std = train_paths.std(dim=(0,1))

    train_dataset = TrajectoryDataset(train_params, train_paths, train_params_mean, train_params_std, train_paths_mean, train_paths_std)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)


    val_params_mean = val_params.mean(dim=0)
    val_params_std = val_params.std(dim=0)

    val_paths_mean = val_paths.mean(dim=(0,1))
    val_paths_std = val_paths.std(dim=(0,1))

    val_dataset = TrajectoryDataset(val_params, val_paths, val_params_mean, val_params_std, val_paths_mean, val_paths_std)
    val_loader = DataLoader(val_dataset, batch_size = val_paths.size(0), shuffle = True)


    # Model Params
    input_size = parameters_data.shape[1]
    hidden_dim = 128 #dimension of the hidden state
    # lstm_hidden_dim = 128
    output_dim = 3
    n_layers = 2
    bidirectional = False
    dropout_rate = 0 #0.5\
    trajectory_length = paths.shape[1]

    model = Dubin_Path_RNN(input_size, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, trajectory_length)

    model.apply(initialize_weights)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # Teacher forcing schedule parameters
    initial_teacher_forcing_prob = 1.0
    final_teacher_forcing_prob = 0.0
    decay_rate = 0.95  # Decay rate per epoch

    writer = SummaryWriter('runs/' + "One_To_Many_LSTM_RNN")

    file_path = "runs/run.csv"
    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        num_epochs = 50
        # teacher_forcing_prob = 1.0
        for epoch in range(num_epochs):
            model.train()
            teacher_forcing_prob = max(final_teacher_forcing_prob, initial_teacher_forcing_prob * (decay_rate ** epoch))
            print(f'Teacher forcing prob {teacher_forcing_prob}')

            for train_params, train_paths in train_loader:
                batch_size = train_params.size(0)
                hidden_in = (torch.zeros(n_layers, batch_size, hidden_dim),
                    torch.zeros(n_layers, batch_size, hidden_dim))
                predictions = model(train_params, train_paths, teacher_forcing_prob, hidden_in)

                # # Reinstate 
                # predictions = predictions * train_paths_std + train_paths_mean
                # train_paths = train_paths * train_paths_std + train_paths_mean

                # Compute train loss
                loss = criterion(predictions, train_paths)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            writer.add_scalar('training loss', loss.item(), epoch)
            with torch.no_grad():
                for val_params, val_paths in val_loader:
                    batch_size = val_params.size(0)
                    hidden_in = (torch.zeros(n_layers, batch_size, hidden_dim),
                                torch.zeros(n_layers, batch_size, hidden_dim))
                    predictions_val = model(val_params, val_paths, teacher_forcing_prob = 0, hidden_in = hidden_in)
                    val_loss = criterion(predictions_val, val_paths)
                writer.add_scalar('val loss', val_loss.item(), epoch)
                
        csv_writer.writerows([predictions[1, :, :].detach().numpy(),train_paths[1, :, :].numpy(), predictions_val[1,:,:].detach().numpy(),val_paths[1,:,:].numpy()])



    # # Normalization of paths (We don't need)
    # # normalized_train_paths = (train_paths - torch.mean(train_paths))/torch.std(train_paths)
    # # normalized_val_paths = (val_paths - torch.mean(val_paths))/torch.std(val_paths)


    # # train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
    # # # tokenizer = get_tokenizer("basic_english")


    # # train_data = train_data.map(
    # #     tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    # # )
    # # test_data = test_data.map(
    # #     tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    # # )



    # # test_size = 0.25

    # # train_valid_data = train_data.train_test_split(test_size=test_size)
    # # train_data = train_valid_data["train"]
    # # valid_data = train_valid_data["test"]



    # # min_freq = 5
    # # special_tokens = ["<unk>", "<pad>"]

    # # # vocab = build_vocab_from_iterator(
    # # #     train_data["tokens"],
    # # #     min_freq=min_freq,
    # # #     specials=special_tokens,
    # # # )


    # # unk_index = vocab["<unk>"]
    # pad_index = 0

    # # # any token not found in the vocabulary will be mapped to the <unk> token.
    # # vocab.set_default_index(unk_index)
    # # # converting each example's tokens to their corresponding numerical indices using the provided vocabulary.
    # # train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    # # valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    # # test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    # # # sets the format of the datasets to PyTorch tensors. It specifies that the columns to be converted to tensors are ids, label, and length
    # # train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
    # # valid_data = valid_data.with_format(type="torch", columns=["ids", "label", "length"])
    # # test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])


    # # train_data[0]

    # batch_size = 512

    # train_data_loader = get_data_loader(train_paths, batch_size, pad_index, shuffle=True)
    # valid_data_loader = get_data_loader(val_paths, batch_size, pad_index)
    # test_data_loader = get_data_loader(val_paths, batch_size, pad_index)

    # # Model Params
    # input_size = parameters_data.shape[1]
    # hidden_dim = 250 #dimension of the hidden state
    # output_dim = 3
    # n_layers = 2
    # bidirectional = False
    # dropout_rate = 0.5

    # model = Multi_Layer_LSTM(input_size, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate)

    # # print(f"The model has {count_parameters(model):,} trainable parameters")

    # model.apply(initialize_weights)
    # # vectors = torchtext.vocab.GloVe()
    # # pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    # # model.embedding.weight.data = pretrained_embedding
    # lr = 5e-4
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # criterion = criterion.to(device)


    # n_epochs = 10
    # best_valid_loss = float("inf")

    # metrics = collections.defaultdict(int)

    # for epoch in range(n_epochs):
    #     train_loss, train_acc = train(
    #         train_data_loader, model, criterion, optimizer, device
    #     )
    #     valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    #     metrics["train_losses"].append(train_loss)
    #     metrics["train_accs"].append(train_acc)
    #     metrics["valid_losses"].append(valid_loss)
    #     metrics["valid_accs"].append(valid_acc)
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), "lstm.pt")
    #     print(f"epoch: {epoch}")
    #     print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    #     print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")




    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(metrics["train_losses"], label="train loss")
    # ax.plot(metrics["valid_losses"], label="valid loss")
    # ax.set_xlabel("epoch")
    # ax.set_ylabel("loss")
    # ax.set_xticks(range(n_epochs))
    # ax.legend()
    # ax.grid()



    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(metrics["train_accs"], label="train accuracy")
    # ax.plot(metrics["valid_accs"], label="valid accuracy")
    # ax.set_xlabel("epoch")
    # ax.set_ylabel("loss")
    # ax.set_xticks(range(n_epochs))
    # ax.legend()
    # ax.grid()

    # model.load_state_dict(torch.load("lstm.pt"))

    # test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)

    # print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

    # # text = "This film is terrible!"

    # # predict_sentiment(text, model, tokenizer, vocab, device)

    # # text = "This film is great!"

    # # predict_sentiment(text, model, tokenizer, vocab, device)
    # # text = "This film is not terrible, it's great!"

    # # predict_sentiment(text, model, tokenizer, vocab, device)

    # # text = "This film is not great, it's terrible!"

    # # predict_sentiment(text, model, tokenizer, vocab, device)

