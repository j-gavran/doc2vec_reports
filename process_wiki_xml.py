import multiprocessing
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pandas_read_xml as pdx
from cleantext import clean
from nltk import sent_tokenize
from sklearn.utils import shuffle
from tqdm import tqdm

COMP_LINK = re.compile(r"(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+")
COMP_NUM = re.compile(r"[0-9]+")


def initial_clean(sent):
    return clean(sent, fix_unicode=True, to_ascii=True)


def remove_punctuation(sent):
    sent_nolink = COMP_LINK.sub("link", sent)
    punct = r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' + "’`”“"
    text_nopunct = "".join(char for char in sent_nolink if char not in punct)
    return text_nopunct


def replace_numbers(sent, replace_with="number"):
    text_nonum = COMP_NUM.sub(replace_with, sent)
    return text_nonum


def word_by_word(sent):
    tokens = re.split(r"\W+", sent[0])
    tokens_no_empty = [word.lower() for word in tokens if word != ""]
    return tokens_no_empty


def word2vec_input(content, max_count=15, min_count=1, min_sent_count=3, drop_first_sent=3, drop_last_sent=6):
    """Prepares content for word2vec.

    Parameters
    ----------
    content: str
        Content of the article.
    max_count: int
        Max word length.
    min_count: int
        Min word length.
    min_sent_count: int
        Min sent length. Drop if less sent.
    drop_first_sent: int
        Drop first n sents.
    drop_last_sent: int
        Drop last n sents.

    Notes
    -----
    sent_tokenize():  splits into sentences and replaces all links with "link",
    a: replaces numbers with "number", removes punctuation, replaces /n,
    b: lowers,
    c: splits sentence into words
    d: drop short and long words
    e: delete empty strings

    Also not take into account short sents.

    Returns
    -------
    Sentences word by word in list: [[w1, w2, ...], [w1, w2, ...], ...]

    """
    tokenizer = initial_clean(content)

    text_in_sent = []

    for sent in sent_tokenize(tokenizer):
        a = [replace_numbers(remove_punctuation(sent)).replace("\n", " ")]
        b = [i.lower() for i in a]
        c = word_by_word(b)
        d = [i if min_count < len(i) < max_count else "" for i in c]
        e = " ".join(d).split()

        if len(e) < min_sent_count:
            continue

        text_in_sent.append(e)

    if drop_last_sent > 0:
        text_in_sent = text_in_sent[drop_first_sent:-drop_last_sent]
    else:
        text_in_sent = text_in_sent[drop_first_sent:]

    return text_in_sent


def doc2vec_input(content, **kwargs):
    w2v_input = word2vec_input(content, **kwargs)

    doc = []
    for sent in w2v_input:
        doc += sent

    return doc


def make_df_from_xml(mediawiki, lock, pos):
    L = len(mediawiki)

    text_df = pd.DataFrame(columns=["id", "title", "timestamp", "bytes", "d2v_text"])

    with lock:
        bar = tqdm(desc=f"process {pos}", total=L, position=pos, leave=False)

    c = 0
    for i in range(L):
        with lock:
            bar.update(1)

        article = mediawiki[i]
        try:
            w2v_text = word2vec_input(article["revision"]["text"]["#text"])

            if len(w2v_text) == 0:
                continue

            text_df.loc[c] = [
                article["id"],
                article["title"],
                article["revision"]["timestamp"],
                article["revision"]["text"]["@bytes"],
                w2v_text,
            ]
            c += 1

        except Exception as e:
            pass

    with lock:
        bar.close()

    return text_df


def df_from_xml_worker(mediawikis, workers=1, **kwargs):
    # https://dumps.wikimedia.org/slwiki/20231001/
    mediawikis = shuffle(mediawikis)

    mediawikis_splits = np.array_split(mediawikis, workers)

    mediawikis = None

    m = multiprocessing.Manager()
    lock = m.Lock()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i, mediawiki in enumerate(mediawikis_splits):
            futures.append(executor.submit(make_df_from_xml, mediawiki, lock, i + 1, **kwargs))

    results = []
    for future in futures:
        results.append(future.result())

    results = pd.concat(results).reset_index(drop=True)

    return results


def pickle_save(path, name, obj):
    with open(path + name, "wb") as f:
        pickle.dump(obj, f)
    return obj


def pickle_load(path, name):
    with open(path + name, "rb") as f:
        obj = pickle.load(f)
    return obj


if __name__ == "__main__":
    wiki_df = pdx.read_xml("slwiki-20231001-pages-articles-multistream.xml", root_is_rows=False).T
    results = df_from_xml_worker(wiki_df["page"]["mediawiki"], workers=16)

    print("saving...")
    pickle_save("./", "wiki_df.pkl", results)
