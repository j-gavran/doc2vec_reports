import logging

import gensim.models as g

from process_wiki_xml import pickle_load

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def for_d2v(sents):
    doc = []
    for sent in sents:
        doc += sent
    return doc


def read_corpus(fname, tokens_only=False):
    for i, tokens in enumerate(fname):
        if tokens_only:
            yield tokens
        else:
            yield g.doc2vec.TaggedDocument(tokens, [i])


def train(
    docs,
    vector_size=300,
    window_size=15,
    min_count=10,
    negative_size=5,
    train_epoch=150,
    dm=0,
    sample=1e-5,
    shrink_windows=False,
    worker_count=23,
    saved_path="./model.bin",
    **kwargs,
):
    train_corpus = list(read_corpus(docs))

    model = g.doc2vec.Doc2Vec(
        vector_size=vector_size,
        min_count=min_count,
        epochs=train_epoch,
        window=window_size,
        negative=negative_size,
        dm=dm,
        shrink_windows=shrink_windows,
        sample=sample,
        workers=worker_count,
        **kwargs,
    )

    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(saved_path)

    return model


if __name__ == "__main__":
    # https://radimrehurek.com/gensim/models/doc2vec.html
    from parse_reports import get_pdf_dct

    use_wiki = True

    if use_wiki:
        df = pickle_load("./", "wiki_df.pkl")
        model_in = list(df["d2v_text"])
    else:
        parsed = get_pdf_dct(path="reports/", full_text=False)
        model_in = list(parsed.values())

    docs = []
    for text in model_in:
        docs.append(for_d2v(text))

    train(docs, saved_path="./model_reco_w15.bin")
