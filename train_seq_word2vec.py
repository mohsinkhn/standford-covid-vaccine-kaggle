from collections import defaultdict
import logging

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from pathlib import Path

from constants import FilePaths


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_6_trids(k=6):
    nucle_com = []
    chars = ["A", "C", "G", "U"]
    base = len(chars)
    end = len(chars) ** k
    for i in range(0, end):
        nuc = ""
        n = i
        for j in range(k):
            ch0 = chars[int(n % base)]
            nuc += ch0
            n /= base
        nucle_com.append(nuc)
        # n = n / base
        # ch1 = chars[int(n % base)]
        # n = n / base
        # ch2 = chars[int(n % base)]
        # n = n / base
        # ch3 = chars[int(n % base)]
        # n = n / base
        # ch4 = chars[int(n % base)]
        # n = n / base
        # ch5 = chars[int(n % base)]
        # nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return nucle_com


def get_4_nucleotide_composition(tris, seq):
    tri_feature = []
    k = len(tris[0])
    for x in range(len(seq) + 1 - k):
        kmer = seq[x : x + k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(str(ind))
    return tri_feature


def make_sentences(sequences, k=5):
    tris = get_6_trids(k=k)
    return [get_4_nucleotide_composition(tris, seq) for seq in sequences]


def avg_nu_emb(seq, model, k=5):
    tris = get_6_trids(k=k)
    k = len(tris[0])
    idx_to_vecs = defaultdict(list)
    for i in range(len(seq)-k+1):
        kmer = str(tris.index(seq[i:i+k]))
        vec = model.wv[kmer]
        for j in range(i, i+k):
            idx_to_vecs[j].append(vec)
    
    embeds = []
    for k, arrs in idx_to_vecs.items():
        embed = np.mean(arrs, 0)
        embeds.append(embed)
    return np.vstack(embeds)


if __name__ == "__main__":
    FP = FilePaths("data")
    train = pd.read_json(FP.train_json, lines=True)
    test = pd.read_json(FP.test_json, lines=True)

    MIN_COUNT = 1
    WINDOW_SIZE = 31
    VECTOR_SIZE = 300
    SAMPLE = 5e-5
    ITER = 100
    K = 5

    Path("data/w2v_embeddings").mkdir(exist_ok=True)
    ids = train.id.tolist() + test.id.tolist()
    sequences = train.sequence.tolist() + test.sequence.tolist()
    all_sentences = make_sentences(sequences, k=K)

    w2v_model = Word2Vec(all_sentences, min_count=MIN_COUNT,
        window=WINDOW_SIZE, size=VECTOR_SIZE, sample=SAMPLE,
        workers=8, iter=ITER, batch_words=100, compute_loss=True)

    #w2v_model.build_vocab(all_sentences, progress_per=10000)
    #w2v_model.train(all_sentences, total_examples=w2v_model.corpus_count, epochs=10, report_delay=1, compute_loss=True)
    w2v_model.wv.save("data/w2v_seq_6gram.vectors")

    for idx, seq in zip(ids, sequences):
        embed_vec = avg_nu_emb(seq, w2v_model, k=K)
        np.save(f"data/w2v_embeddings/{idx}.npy", embed_vec)
    
    print(np.mean(embed_vec), np.median(embed_vec), np.std(embed_vec), np.max(embed_vec), np.min(embed_vec))