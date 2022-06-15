from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair
        )
        self.sentences = [transform([data[sent_idx]]) for data in dataset]
        self.labels = [np.int32(data[label_idx]) - 10 for data in dataset]

    def __getitem__(self, item):
        return self.sentences[item] + (self.labels[item],)

    def __len__(self):
        return len(self.labels)


class BERTDatasetInference(Dataset):
    def __init__(self, sentence, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair
        )
        self.sentences = [transform(sentence)]

    def __getitem__(self, item):
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)
