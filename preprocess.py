import jieba
import pandas as pd


def distinct_star(sent):
    new_sent = ""
    prev_is_star = False
    for ch in sent:
        if prev_is_star:
            if ch == "*":
                continue
            else:
                prev_is_star = False
        else:
            if ch == "*":
                prev_is_star = True
        new_sent += ch

    return new_sent


def cut(sent_list):
    sent_list_by_word = []
    i = 0
    for sent in sent_list:
        sent = distinct_star(sent)
        words = jieba.cut(sent)
        sent_list_by_word.append(list(words))
        if i%100 == 0:
            print("{} sents parsed!".format(i))
        i += 1

    return sent_list_by_word


def merge_ham_spam():
    ham_file = "./data/ham.txt"
    spam_file = "./data/spam.txt"

    # load from txt
    import load
    ham_sent_list = cut(load.load_text(ham_file))
    spam_sent_list = cut(load.load_text(spam_file))

    # create DataFrame
    # ham label 0
    # spam label 1
    ham_df = pd.DataFrame(pd.Series(ham_sent_list))
    ham_df.columns = ["sent"]
    spam_df = pd.DataFrame(pd.Series(spam_sent_list))
    spam_df.columns = ["sent"]

    ham_df["label"] = 0
    spam_df["label"] = 1

    # merge
    df = pd.concat([ham_df, spam_df])
    # df.to_pickle("./data/sent_with_label.pkl")
    return df


def create_dict(df):
    all_words = set()
    for sent in df["sent"]:
        for w in sent:
            all_words.add(w)

    word2id = dict()
    for word_id, w in enumerate(all_words):
        word2id[w] = word_id + 1  # skip 0 for unknown word

    # pd.to_pickle(word2id, "./data/word2id")
    return word2id


def sent_list_to_id(df:pd.DataFrame, word2id):
    def row_func_to_id(row):
        sent = row["sent"]
        id_sent = []
        for w in sent:
            id_sent.append(word2id[w])
        return id_sent

    df["id_sent"] = df.apply(row_func_to_id, axis=1)

    # df.to_pickle("./data/id_sent.pkl")
    return df