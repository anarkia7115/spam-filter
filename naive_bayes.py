import pandas as pd


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


def main():
    import load
    df = load.load_id_sent()
    return word_document_count(df)
