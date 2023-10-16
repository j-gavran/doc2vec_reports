import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def get_cos_sim(pdfs_lst, idx, model, lock, pos, degs=True):
    i_idx, j_idx = idx
    res_cos, idx_names = [], []

    with lock:
        bar = tqdm(total=len(i_idx), position=pos, leave=False, desc=f"started process {pos}")

    for i, j in zip(i_idx, j_idx):
        name_i, text_i = pdfs_lst[0][i], pdfs_lst[1][i]
        sim_i = model.infer_vector(text_i).reshape(1, -1)

        name_j, text_j = pdfs_lst[0][j], pdfs_lst[1][j]
        sim_j = model.infer_vector(text_j).reshape(1, -1)

        if degs:
            cos = np.degrees(np.arccos(cosine_similarity(sim_i, sim_j)[0][0]))
        else:
            cos = cosine_similarity(sim_i, sim_j)[0][0]

        res_cos.append(cos)
        idx_names.append([(i, name_i), (j, name_j)])

        with lock:
            bar.update(1)
            bar.set_description(f"{name_i} vs {name_j}")

    return res_cos, idx_names


def get_vectors(pdfs_dct, vec_size, model):
    vec_mat = np.zeros((len(pdfs_dct), vec_size))

    for i, text in tqdm(enumerate(pdfs_dct.values()), desc="Getting doc2vec vectors"):
        vec_mat[i, :] = model.infer_vector(text).reshape(1, -1)

    return vec_mat


def cos_sim_worker(pdfs_dct, model, workers=1, **kwargs):
    n = len(pdfs_dct)

    i_idx, j_idx = np.triu_indices(n, k=1)
    i_idx_split, j_idx_split = np.array_split(i_idx, workers), np.array_split(j_idx, workers)

    m = multiprocessing.Manager()
    lock = m.Lock()

    pdfs_lst = [list(pdfs_dct.keys()), list(pdfs_dct.values())]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i, (i_split, j_split) in enumerate(zip(i_idx_split, j_idx_split)):
            futures.append(executor.submit(get_cos_sim, pdfs_lst, [i_split, j_split], model, lock, i + 1, **kwargs))

    results = []
    for future in futures:
        results.append(future.result())

    cos_mat = np.zeros((n, n))
    nan_idx = np.tril_indices(n, k=1)
    cos_mat[nan_idx] = np.nan

    cos_vals, name_pairs = [], []

    for res in results:
        cos, idx_names = res
        for k, ((i, name_i), (j, name_j)) in enumerate(idx_names):
            cos_mat[i, j] = cos[k]
            cos_vals.append(cos[k])
            name_pairs.append((name_i, name_j))

    return cos_mat.T, [cos_vals, name_pairs]


def censor_name(full_name):
    split_full_name = full_name.split("_")
    censor_name = ""
    for name in split_full_name:
        censor_name += name[0]
    return censor_name


def plot_cos_corr_matrix(pdfs_dct, res_cos, save=False, scale=90, annot=False, censor=False, **kwargs):
    plt.figure(figsize=(12, 10))

    if annot:
        annot = {"annot": True, "fmt": ".2f", "annot_kws": {"size": 35 / np.sqrt(len(res_cos))}}
    else:
        annot = {}

    if censor:
        xticklabels = [censor_name(name) for name in list(pdfs_dct.keys())]
        yticklabels = [censor_name(name) for name in list(pdfs_dct.keys())]
    else:
        xticklabels = list(pdfs_dct.keys())
        yticklabels = list(pdfs_dct.keys())

    sns.heatmap(
        1 - res_cos / scale if scale is not None else res_cos,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap="viridis",
        cbar_kws={"label": "Cosine similarity"},
        **annot,
        **kwargs,
    )
    plt.tight_layout()

    if save:
        plt.savefig(save)

    plt.show()
    plt.close()


def plot_cos_scores(cos_vals, name_pairs, top=-1, reverse=True, thres=None, save=False, censor=False):
    cos_vals = np.array(cos_vals)
    sorted_idx = np.argsort(cos_vals)

    if reverse:
        sorted_idx = sorted_idx[::-1]

    for idx in sorted_idx[:top]:
        print(f"{name_pairs[idx]}: {cos_vals[idx]:.3f}")

    if censor:
        labels = np.array([f"{censor_name(name_i)} - {censor_name(name_j)}" for name_i, name_j in name_pairs], dtype=object)
    else:
        labels = np.array([f"{name_i} - {name_j}" for name_i, name_j in name_pairs], dtype=object)

    fig, ax = plt.subplots(figsize=(8, 10))

    x = np.arange(0, len(cos_vals), 1)[:top]

    ax.scatter(cos_vals[sorted_idx][:top], x)
    ax.set_yticks(x)
    ax.set_yticklabels(labels[sorted_idx][:top], rotation=0, fontsize=10)
    ax.set_xlabel("Cosine similarity", fontsize=12)

    if thres is not None:
        lin_func = lambda x, a, b: a * x + b

        y = cos_vals[sorted_idx][:top]
        y_idx = np.argwhere(y < thres).flatten()

        popt, pcov = curve_fit(lin_func, x[y_idx], y[y_idx])
        ax.plot(lin_func(x, *popt), x, "r--")

    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save)

    plt.show()
    plt.close()


def get_2d_pca(pdfs_dct, res_vec, save=False, censor=False):
    pca = PCA(n_components=2)
    proj_2d = pca.fit_transform(res_vec)

    plt.figure(figsize=(10, 8))

    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], alpha=0.8)

    for i, (x, y) in enumerate(zip(proj_2d[:, 0], proj_2d[:, 1])):
        if censor:
            plt.text(x, y, censor_name(list(pdfs_dct.keys())[i]), fontsize=10, alpha=0.9)
        else:
            plt.text(x, y, list(pdfs_dct.keys())[i], fontsize=10, alpha=0.9)

    plt.tight_layout()
    if save:
        plt.savefig(save)

    plt.show()
    plt.close()


if __name__ == "__main__":
    import gensim.models as g

    from parse_reports import get_pdf_dct

    vec_size = 300

    pdfs_dct = get_pdf_dct(
        path="reports/",
        ignore=None,
        engine="textract",
        skip=500,
    )

    model = g.Doc2Vec.load("model_reco.bin")  # doc2vec model

    censor = True

    res_cos, scores = cos_sim_worker(pdfs_dct, model, workers=10, degs=False)
    plot_cos_corr_matrix(pdfs_dct, res_cos, scale=None, censor=censor, save="cos_sim.png")
    plot_cos_scores(*scores, top=50, thres=0.65, censor=censor, save="cos_sim_scores.png")

    res_vec = get_vectors(pdfs_dct, vec_size, model)
    get_2d_pca(pdfs_dct, res_vec, censor=censor, save="pca.png")
