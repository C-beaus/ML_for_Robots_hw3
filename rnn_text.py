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
from mpl_toolkits.mplot3d import Axes3D


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
        self.output_layer = nn.Sequential(
             nn.Linear(hidden_dim, 256),
             nn.ReLU(),
             nn.Linear(256, output_dim)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, initial_conditions, ground_truth, teacher_forcing_prob, hidden_in):
       
       batch_size = initial_conditions.size(0)
       lstm_input = self.fc(initial_conditions).unsqueeze(1)

       outputs = []

       for i in range(ground_truth.size(1)):
           out, hidden_in = self.lstm(lstm_input, hidden_in)
           trajectory_point = self.output_layer(out.squeeze(1))
           outputs.append(trajectory_point.unsqueeze(1))

           use_teacher_forcing = torch.rand(1).item() < teacher_forcing_prob
           if use_teacher_forcing and ground_truth is not None:
                lstm_input = ground_truth[:, i].unsqueeze(1)
           else:
                lstm_input = trajectory_point.unsqueeze(1)
       return torch.cat(outputs, dim=1)
    

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
    paths = np.zeros((10000, 175, 3))

    df = sio.loadmat(data_path)
    df = df['master_path']
    # Loop through all files in the folder
    for i in range(df.shape[2]):
        if i == 0:
            paths[i, :, :] = df[0:175, :, i]
        else:
            paths[i, :, :] = df[0:175, :, i]
    paths = torch.tensor(paths, dtype=torch.float32).to(device)
    
    train_indices = int(((i + 1) * 0.8) - 1)

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
    output_dim = 3
    n_layers = 2
    bidirectional = False
    dropout_rate = 0.0 # 0.5
    trajectory_length = paths.shape[1]

    model = Dubin_Path_RNN(input_size, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, trajectory_length)

    model.apply(initialize_weights)
    lr = 1e-3 #1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # Teacher forcing schedule parameters
    initial_teacher_forcing_prob = 1.0
    final_teacher_forcing_prob = 0.0
    decay_rate = 0.95 #0.95  # Decay rate per epoch

    writer = SummaryWriter('runs/' + "One_To_Many_LSTM_RNN")

    # Create image folder
    # Will's Path
    # folder_path = "C:/Users/willi/Documents/GitHub/ML_for_Robots_hw3/images"
    # Chase's Path
    folder_path = 'c:/Users/chase/OneDrive/Documents/Grad/ML_for_Robots/hw_3/ML_for_Robots_hw3/images'
    os.makedirs(folder_path, exist_ok=True)

    file_path = "runs/run.csv"
    with open(file_path, mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        num_epochs = 20
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
            
            if epoch == 1 | epoch == 10:
                # Create a 3D plot
                for d in range(10):
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')

                    x_pred = predictions_val[d, :, 0].detach().numpy()  
                    y_pred = predictions_val[d, :, 1].detach().numpy()  
                    z_pred = predictions_val[d, :, 2].detach().numpy()  

                    x_gtruth = val_paths[d, :, 0].numpy()  
                    y_gtruth = val_paths[d, :, 1].numpy()  
                    z_gtruth = val_paths[d, :, 2].numpy()  

                    ax.plot(x_pred, y_pred, z_pred, label=f'Prediction, Depth {d+1}', marker='o')
                    ax.plot(x_gtruth, y_gtruth, z_gtruth, label=f'Ground Truth, Depth {d+1}', marker='_')

                    img_path = f'images/test_img_epoch_{epoch}_{d+1}.png'

                    # Set labels
                    ax.set_xlabel('X Axis')
                    ax.set_ylabel('Y Axis')
                    ax.set_zlabel('Z Axis')
                    plt.title(f'3D Plot of Trajectory Data for Path {d+1}')
                    plt.legend()
                    plt.savefig(img_path)
        
                
        csv_writer.writerows([predictions[1, :, :].detach().numpy(),train_paths[1, :, :].numpy(), predictions_val[1,:,:].detach().numpy(),val_paths[1,:,:].numpy()])

    # Create a 3D plot
    for d in range(10):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        x_pred = predictions_val[d, :, 0].detach().numpy()  # Feature 1
        y_pred = predictions_val[d, :, 1].detach().numpy()  # Feature 2
        z_pred = predictions_val[d, :, 2].detach().numpy()  # Feature 3

        x_gtruth = val_paths[d, :, 0].numpy()  # Feature 1
        y_gtruth = val_paths[d, :, 1].numpy()  # Feature 2
        z_gtruth = val_paths[d, :, 2].numpy()  # Feature 3

        ax.plot(x_pred, y_pred, z_pred, label=f'Prediction, Depth {d+1}', marker='o')
        ax.plot(x_gtruth, y_gtruth, z_gtruth, label=f'Ground Truth, Depth {d+1}', marker='_')

        img_path = f'images/test_img{d+1}.png'

        # Set labels
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title(f'3D Plot of Trajectory Data for Path {d+1}')
        plt.legend()
        plt.savefig(img_path)
    plt.show()