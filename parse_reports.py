import logging
import operator
import os

import pypdfium2 as pdfium
import textract
from tqdm import tqdm

from process_wiki_xml import doc2vec_input

REPLACE = {
    "\x14c": "c",
    "\x14z": "z",
    "\x14s": "s",
    "�": "",
    "\x16": "",
    "ˇ": "",
    "\n": " ",
    "²": "s",
    "£": "c",
    "\x83": "c",
    "\x92": "s",
    "-": "",
    "—": "",
    "'": "",
    "￾": "",
}


def read_pdf(path, engine="pdfium", skip=1):
    # more accurate, slower
    if engine == "textract": 
        full_text = textract.process(path, method="pdfminer").decode("utf-8")[skip:]

    # less accurate, faster
    elif engine == "pdfium": 
        pdf = pdfium.PdfDocument(path)
        pages = [page.get_textpage().get_text_range() for page in pdf]
        full_text = "".join(pages[skip:])

    else:
        raise ValueError(f"Invalid engine: {engine}")

    for k, v in REPLACE.items():
        full_text = full_text.replace(k, v)

    return full_text


def get_pdf_dct(
    path="reports/",
    remove_nums=True,
    remove_links=True,
    ignore=None,
    full_text=False,
    **kwargs,
):
    pdfs = os.listdir(path)

    if ignore is None:
        ignore = []

    pdfs_dct, full_text_dct = dict(), dict()
    for pdf in tqdm(pdfs, desc="Reading pdfs"):
        if pdf in ignore:
            logging.warning(f"ignoring {pdf}")
            continue
        
        if ".pdf" not in pdf:
            continue

        text = read_pdf(path + pdf, **kwargs)
        d2v = doc2vec_input(text, min_count=1, drop_first_sent=0, drop_last_sent=0)

        if remove_nums:
            d2v = [i for i in d2v if "number" not in i]

        if remove_links:
            d2v = [i for i in d2v if "link" not in i]

        name = "_".join(pdf.split("_")[1:])[:-4]

        if full_text:
            full_text_dct[name] = text

        pdfs_dct[name] = d2v

    if full_text:
        return pdfs_dct, full_text_dct
    else:
        return pdfs_dct


def count_words(tokenized_lst, sort=True):
    word_count = {}

    for word in tokenized_lst:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 0

    if sort:
        return sorted(word_count.items(), key=operator.itemgetter(1))
    else:
        return word_count


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parsed, text = get_pdf_dct(path="reports/", full_text=True, engine="textract", skip=500)

    with open("reports.txt", "w") as fp:
        for name, txt in text.items():
            fp.write(f"{name}")
            fp.write(f"\n\n{txt}\n\n")
            fp.write(f"\n\n{parsed[name]}\n\n")

    flat_parsed = [item for sublist in list(parsed.values()) for item in sublist]
    counts = count_words(flat_parsed)

    for k, v in counts[-50:]:
        plt.bar(k, v, color="C0")
        plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()
