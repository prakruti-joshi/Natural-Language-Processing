import math
import sys

from collections import Counter
from functools import reduce


def compute_bleu(reflists, hyps, n_max=4, use_shortest_ref=False):
    assert len(reflists) == len(hyps)
    prec_mean = 0  # TODO: Implement
    brevity_penalty = 0  # TODO:Implement
    an_sum = [0]*n_max
    bn_sum = [0]*n_max
    H = 0
    R = 0

    # Looping over l= 1..N
    for l in range(len(reflists)):
        refs = reflists[l]  # Reference list
        hyp = hyps[l]       # Hypothesis sentence

        # Calculting a_n and b_n
        for n in range(1, n_max+1):
            an, bn = get_ngram_counts(refs, hyp, n)
            an_sum[n-1] += an
            bn_sum[n-1] += bn
        # Calculating H
        H += len(hyp)
        # Calculating R
        diff = math.inf
        r_len = 0
        for j in range(len(refs)):
            l_diff = abs(len(refs[j])-len(hyp))
            if l_diff<diff:
                r_len = len(refs[j])
        R += r_len

    brevity_penalty = min(1.0, math.exp(float(1-(float(R)/H))))
    p_n = 1.0
    for n in range(n_max):
        p = float(an_sum[n]/bn_sum[n])
        p_n *= p
    prec_mean = p_n**(1/n_max)
    bleu = float(brevity_penalty * prec_mean)
    return bleu


def get_ngram_counts(refs, hyp, n):
    hyp_ngrams = [tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)]
    num_hyp_ngrams = max(1, len(hyp_ngrams))  # Avoid empty

    num_hyp_ngrams_in_refs_clipped = 0  # TODO: Implement

    Ngrams = Counter(hyp_ngrams)
    for g,c in Ngrams.items():
        c_max = -math.inf
        for j in range(len(refs)):
            ref_grams = [tuple(refs[j][i:i + n]) for i in range(len(refs[j]) - n + 1)]
            ref_grams_count = Counter(ref_grams)
            g_count = ref_grams_count[g]
            c_max = max(g_count, c_max)
        a_n = min(c_max, c)
        num_hyp_ngrams_in_refs_clipped += a_n

    return num_hyp_ngrams_in_refs_clipped, num_hyp_ngrams

