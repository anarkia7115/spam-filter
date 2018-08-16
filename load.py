import pandas as pd


def load_text(text_file):
    sent_list = []
    with open(text_file, 'r') as f:
        for content in f:
            content = content.strip().lstrip("[\"")
            content = content.rstrip("\"]")
            content = content.replace("\",\"", " ")
            sent_list.append(content)

    return sent_list


def load_df():
    return pd.read_pickle("./data/sent_with_label.pkl")


def load_word_dict():
    return pd.read_pickle("./data/word2id")


def load_id_sent():
    return pd.read_pickle("./data/id_sent.pkl")


def load_word_prob():
    return pd.read_pickle("./data/word2label2prob")


def load_test_df():
    return pd.read_pickle("./data/content.pkl")


def main():
    spam_file = "./data/spam.txt"
    sent_list = load_text(spam_file)
    return sent_list
