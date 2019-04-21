import argparse

from string import punctuation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

# 2 words to the left, 2 to the right
CONTEXT_SIZE = 2
EMBEDDING_SIZE = 10


parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--train', action='store_true', help='Train model and generate CSV submission file')
parser.add_argument('-te', '--test', action='store_true', help='Test model and generate CSV submission file')
# luan xu

def process_data():
    df = pd.read_csv('./csv/train.csv', nrows=10)
    comments = []
    vocabs = []
    for comment in df['comment_text']:
        for c in punctuation:
            comment = comment.replace(c, ' ') if c in comment else comment
        vocab = comment.lower().split()
        vocabs = vocab + vocabs
        comments.append(vocab)
    return df['target'], comments, vocabs


class CBOW(nn.Module):
    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        return out


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Part of our loss function will be a penalty term.
# Pearson Correlation requires us to calculate Covariance:
def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def pcor(m, y=None):
    x = cov(m, y)

    # Not interested in positive nor negative correlations;
    # We're only interested in correlation magnitudes:
    x = (x * x).sqrt()  # no negs

    stddev = x.diag().sqrt()
    x = x / stddev[:, None]
    x = x / stddev[None, :]
    return x


def pcor_err(m, y=None):
    # Every var is correlated with itself, so subtract that:
    x = (pcor(m, y) - torch.eye(m.size(0)))
    return x.mean(dim=0).mean()


if __name__ == '__main__':
    print('Process training data...')
    values, comments, vocab = process_data()
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []

    for comment in comments:
        for i in range(CONTEXT_SIZE, len(comment) - CONTEXT_SIZE):
            context = [comment[i - 2], comment[i - 1],
                       comment[i + 1], comment[i + 2]]
            target = autograd.Variable(torch.LongTensor([word_to_ix[comment[i]]]))
            data.append((context, target))
            print(data)

    loss_func = nn.CrossEntropyLoss()

    net = CBOW(CONTEXT_SIZE, embedding_size=EMBEDDING_SIZE, vocab_size=vocab_size)
    optimizer = optim.Adam(net.parameters())


    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            context_var = make_context_vector(context, word_to_ix)
            net.zero_grad()
            probs = net(context_var)

            loss = loss_func(probs.view(1, -1), target) + pcor_err(
                torch.transpose(net.embeddings.weight, 0, 1)) * 6  # (vocab_size//2+1) ?
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        print(total_loss)
    c = np.corrcoef(net.embeddings.weight.data.numpy().T)

    print(type(net.embeddings.weight.data))
    print(type(net.embeddings.weight))
    print(type(net.embeddings))
