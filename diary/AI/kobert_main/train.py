import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

import gluonnlp as nlp
import argparse
import pandas as pd

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers.optimization import get_cosine_schedule_with_warmup

from .data import BERTDataset
from .models import BERTClassifier


# Arguments #
parser = argparse.ArgumentParser(description="Parameters")
parser.add_argument("--max_len", "-ml", type=int, default=256, help="max length of text data")
parser.add_argument("--batch_size", "-b", type=int, default=64, help="batch size")
parser.add_argument(
    "--freeze_layer", "-fl",
    type=int,
    default=0,
    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    help="for hyperparameter tuning choose from what layer to freeze, set 0 to ignore and go full training"
)
parser.add_argument("--warmup_ratio", "-wr", type=float, default=0.1, help="warmup ratio")
parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
parser.add_argument("--max_grad_norm", "-mgn", type=int, default=1, help="gradient norm clipping")
parser.add_argument("--log_interval", "-li", type=int, default=200, help="interval for training log output")
parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--cuda", "-c", type=str, default="gpu", choices=["gpu", "cpu"], help="use which device to train")
parser.add_argument("--model", "-m", type=str, default=None, help="path to the pretrained model")
args = parser.parse_args()


# Parameters #
max_len = args.max_len
batch_size = args.batch_size
freeze_layer = str(args.freeze_layer)
warmup_ratio = args.warmup_ratio
num_epochs = args.epochs
max_grad_norm = args.max_grad_norm
log_interval = args.log_interval
learning_rate = args.learning_rate
cuda = args.cuda
model_path = args.model


# CUDA #
device = torch.device("cuda:0" if torch.cuda.is_available() and cuda == "gpu" else "cpu")
if torch.cuda.is_available() and cuda == "gpu":
    print("Using GPU")
else:
    print("Using CPU")


# Model load #
bert_model, vocab = get_pytorch_kobert_model()
model = BERTClassifier(bert_model, dr_rate=0.5).to(device)
if model_path:
    model.load_state_dict(torch.load(model_path))

trainable = False
for name, param in model.named_parameters():
    if freeze_layer == "0":
        pass
    else:
        if freeze_layer in name:
            trainable = True
        param.requires_grad = trainable


# Data #
tokenizer = get_tokenizer()
token = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

dataset_train = nlp.data.TSVDataset('~/dataset/training.tsv', field_indices=[0, 1], num_discard_samples=1)
dataset_valid = nlp.data.TSVDataset('~/dataset/validation.tsv', field_indices=[0, 1], num_discard_samples=1)

data_train = BERTDataset(dataset_train, 0, 1, token, max_len, True, False)
data_valid = BERTDataset(dataset_valid, 0, 1, token, max_len, True, False)

dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=4)
dataloader_valid = DataLoader(data_valid, batch_size=batch_size, num_workers=4)


# Optimizer and Schedule #
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = CrossEntropyLoss()

t_total = len(dataloader_train) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# Accuracy #
def calculate_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


# Train #
losses_train, accuracies_train = [], []
losses_valid, accuracies_valid = [], []

for epoch in range(num_epochs):
    accuracy_train = 0.
    accuracy_valid = 0.

    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader_train):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = criterion(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        accuracy_train += calculate_accuracy(out, label)

        if batch_id % log_interval == 0:
            print(f"{epoch + 1:>{len(str(num_epochs))}}/{num_epochs} batch ID {batch_id + 1:>4}", end=" ")
            print((
                f"loss {loss.data.cpu().numpy():>1.3f} "
                f"accuracy {accuracy_train / (batch_id + 1):>1.3f}"
            ), end="\n")
    print((
        f"\nEpoch {epoch + 1} "
        f"train loss {loss.data.cpu().numpy():>1.3f} "
        f"train accuracy {accuracy_train / (batch_id + 1):>1.3f}"
    ))
    losses_train.append(loss.data.cpu().numpy())
    accuracies_train.append(accuracy_train)

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader_valid):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss_val = criterion(out, label)
        accuracy_valid += calculate_accuracy(out, label)

    print((
        f"Epoch {epoch + 1} "
        f"valid loss {loss_val.data.cpu().numpy():>1.3f} "
        f"valid accuracy {accuracy_valid / (batch_id + 1):>1.3f}\n"
    ))
    losses_valid.append(loss_val.data.cpu().numpy())
    accuracies_valid.append(accuracy_valid)

# Model Save #
torch.save(model.state_dict(), "model.pt")


# Log Save #
d = {
    'training_loss': losses_train,
    'training_accuracy': accuracies_train,
    'validation_loss': losses_valid,
    'validation_accuracy': accuracies_valid
}
DF = pd.DataFrame(data=d)
DF.to_csv('training_log.csv', sep=",", encoding="utf-8")
