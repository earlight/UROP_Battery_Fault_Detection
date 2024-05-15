import copy
import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split

# Load the data
brand = 1
df = pd.read_csv('data/brand%d.csv' % brand)

# Process the data
df = df.dropna().reset_index(drop=True)
# df = df.sample(frac=0.001, random_state=99).reset_index(drop=True) # TODO: small sample for testing
df = df.drop(columns=['car'], axis=1)
df = df.drop(columns=['charge_segment'], axis=1)

# Normalize all columns
for column in df.columns:
    if column != 'label':
        df[column] = (df[column] - df[column].mean()) / df[column].std()

# Seperate input and output data
X = df.drop(['label'], axis=1)
Y = df['label']

# Get number of 0s and 1s in data
print("Number of 0s:", len(Y) - Y.sum())
print("Number of 1s:", Y.sum())
print("Data proportion of 0s:", (len(Y) - Y.sum()) / len(Y), "\n")

X = torch.tensor(X.values, dtype=torch.float32)
Y = torch.tensor(Y.values, dtype=torch.float32).reshape(-1, 1)
print("X tensor:", X.shape)
print("Y tensor:", Y.shape, "\n")

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(9, 9)
        self.layer2 = nn.Linear(9, 9)
        self.layer3 = nn.Linear(9, 9)
        self.layer4 = nn.Linear(9, 9)
        self.act = nn.ReLU()
        self.output = nn.Linear(9, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        x = self.sigmoid(self.output(x))
        return x
    
def model_train(model, X_train, y_train, X_val, y_val):

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.004)

    n_epochs = 96   # number of epochs to run
    batch_size = 64  # size of each batch 
    batch_start = torch.arange(0, len(X_train), batch_size)
    schduler = optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 3 + 1, gamma=0.5)

    # Hold the best model
    best_val_acc = - np.inf   # init to negative infinity
    best_weights = None

    # Track training and validation accuracy
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")

            for start in bar:

                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]

                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # update weights
                optimizer.step()

                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )

        # evaluate training accuracy at end of each epoch
        model.eval()
        y_pred_train = model(X_train)
        train_acc = (y_pred_train.round() == y_train).float().mean()
        train_acc = float(train_acc)
        train_accuracies.append(train_acc)

        # evaluate validation accuracy at end of each epoch
        model.eval()
        y_pred_val = model(X_val)
        val_acc = (y_pred_val.round() == y_val).float().mean()
        val_acc = float(val_acc)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        # update learning rate
        schduler.step()

        print(f"Epoch {epoch} took {time.time() - start_time:.6f}s. Train accuracy: {train_acc:.6f}. Val accuracy: {val_acc:.6f}. LR: {optimizer.param_groups[0]['lr']}")

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("Best validation accuracy:", best_val_acc)
    return train_accuracies, val_accuracies

# Split data into training, validation, and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=88)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create the model
model = Deep()
train_accuracies, val_accuracies = model_train(model, X_train, y_train, X_val, y_val)

# Save the model
model_timestamp = time.strftime("%m-%d_%H-%M-%S")
model_path = './models/v3_less_epochs/results/' + model_timestamp + '_brand' + str(brand) + '.pth'
torch.save(model.state_dict(), model_path)

# Test the model
model.eval()
y_pred_test = model(X_test)
test_acc = (y_pred_test.round() == y_test).float().mean()

# Plot the training and validation accuracy when threshold is 0.5
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy (Threshold=0.5)")
plt.title("Train and Val Accuracy (Test Accuracy: %.6f)" % float(test_acc))
plt.savefig('./models/v3_less_epochs/results/' + model_timestamp + '_brand' + str(brand) + '.png')

# Calculate ROC
y_pred_test = y_pred_test.detach().numpy()
y_test = y_test.numpy()

# Save test data
test_data = pd.DataFrame(data={'y_pred_test': y_pred_test.flatten(), 'y_test': y_test.flatten()})
test_data.to_csv('./models/v3_less_epochs/results/' + model_timestamp + '_brand' + str(brand) + '.csv', index=False)