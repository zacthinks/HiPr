import fasttext
import math
import pandas as pd
from statistics import mean, median
from numpy import cumsum


def combine_tuples(tuples_list):
    d = {}
    for k, v in tuples_list:
        if k in d:
            d[k] += v
        else:
            d[k] = v
    return d


def lang_checker(pages, model, window, number, padding=50):
    def get_pse(cum_lens, pos):
        prev_len = 0
        for i, cum_len in enumerate(cum_lens):
            if pos < cum_len:
                pos_i = pos - prev_len
                i_len = cum_len - prev_len
                diff = cum_len - pos
                if diff > window + padding:
                    return i, pos_i, pos_i + window
                elif diff >= window / 2:
                    return i, max(padding, i_len - padding - window), max(padding, i_len - padding)
                else:
                    if i == len(cum_lens) - 1:
                        return i, max(padding, i_len - padding - window), max(padding, i_len - padding)
                    else:
                        i_len = cum_lens[i + 1] - cum_len
                        return i + 1, min(i_len, padding), min(i_len, padding + window)
            prev_len = cum_len

    if pages is None or len(pages) == 0 or sum([len(page) for page in pages]) == 0:
        return pd.Series(['NO_TEXT', 0, 0, 0, 0],
                         index=['top_lang', 'lang_score_max', 'lang_score_mean', 'lang_score_median', 'top_lang_prop'])

    lens = [len(page) for page in pages]
    cum_lens = cumsum(lens)
    text_len = cum_lens[-1]

    if text_len <= window * number:
        if text_len < 2 * window:
            chunk_num = 1
        else:
            chunk_num = math.floor(text_len / window)
    else:
        chunk_num = number

    chunk_size = text_len / (chunk_num + 1)
    start_locs = [max(0, math.ceil(((x + .5) * chunk_size) - (window / 2))) for x in range(chunk_num)]
    pses = [get_pse(cum_lens, start_loc) for start_loc in start_locs]

    if None in pses:
        print("Failed to get extracts for the following:")
        print(start_locs)
        print(pages)

        return pd.Series(['ERROR', 0, 0, 0, 0],
                         index=['top_lang', 'lang_score_max', 'lang_score_mean', 'lang_score_median', 'top_lang_prop'])

    texts = [pages[p][s:e] for p, s, e in pses]

    try:
        lang_scores = [(p[0][0][9:], p[1][0]) for t in texts if (p := model.predict(t))]
        lang_scores_dict = combine_tuples(lang_scores)
        top_lang = max(lang_scores_dict.items(), key=lambda x: x[1])[0]
        top_lang_scores = [x[1] for x in lang_scores if x[0] == top_lang]

        return pd.Series([top_lang,
                          max(top_lang_scores),
                          mean(top_lang_scores),
                          median(top_lang_scores),
                          (len(top_lang_scores) / len(texts))],
                         index=['top_lang', 'lang_score_max', 'lang_score_mean', 'lang_score_median', 'top_lang_prop'])

    except Exception as e:
        print(type(e).__name__)
        print(e)
