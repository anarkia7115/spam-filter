import pandas as pd
import numpy as np


def word_document_count(df: pd.DataFrame):
    wordlabel2dc = dict()
    def row_func_count(row):
        id_sent = row["id_sent"]
        label = row["label"]
        id_sent = set(id_sent)
        for word_id in id_sent:
            key = (word_id, label)
            if key in wordlabel2dc:
                wordlabel2dc[key] += 1
            else:
                wordlabel2dc[key] = 1
    word2label2dc = dict()

    df.apply(row_func_count, axis=1)

    for (word, label), dc in wordlabel2dc.items():

        if word in word2label2dc:
            word2label2dc[word][label] = dc
        else:
            word2label2dc[word] = {label:dc}

    return word2label2dc


def word_prob(word2label2dc):

    word2label2prob = dict()

    for word, label2dc in word2label2dc.items():
        sum = np.sum(list(label2dc.values()))
        prob = dict()

        if 0 in label2dc:
            prob[0] = label2dc[0] / sum
        else:
            prob[0] = 0

        if 1 in label2dc:
            prob[1] = label2dc[1] / sum
        else:
            prob[1] = 0

        word2label2prob[word] = prob


    return word2label2prob


def main():
    import load
    df = load.load_id_sent()
    word2label2dc = word_document_count(df)
    word2label2prob = word_prob(word2label2dc)

    # pd.to_pickle(word2label2prob, "./data/word2label2prob")
    return word2label2prob
