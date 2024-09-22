import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

import spacy
import re


class InvalidIOB(Exception):
    def __init__(self, position):
        self.position = position
        super().__init__(f"Invalid BIO tag sequence at position {position}.")


def get_child_dict(tok):
    child_dict = {}
    for child in tok.children:
        dep = child.dep_
        if dep in ['punct']:
            continue
        if dep in child_dict:
            child_dict[dep] = child_dict[dep] + "," + child.text
        else:
            child_dict[dep] = child.text
    return child_dict


def get_range_head(doc, start, end, return_conjs=True):
    head_i = start
    while (doc[head_i].head.i > start
           and doc[head_i].head.i <= end
           and doc[head_i].head.i != head_i):
        head_i = doc[head_i].head.i

    if doc[head_i].pos_ in ['ADP', 'SCONJ']:
        for child in doc[head_i].children:
            if child.dep_ == 'pobj':
                head_i = child.i
                break

    quantifiers = ['all', 'some', 'more', 'lot', 'lots', 'enough', 'none', 'any', 'most', 'less', 'much']
    if (doc[head_i].text.lower() in quantifiers) and ('of' in [c.text.lower() for c in doc[head_i].children]):
        of_tok = next(c for c in doc[head_i].children if c.text.lower() == 'of')
        head_i = next((c.i for c in of_tok.children if c.dep_ == 'pobj'), head_i)

    head_info = [(head_i, doc[head_i].text, doc[head_i].lemma_, doc[head_i].ent_type_, get_child_dict(doc[head_i]))]
    if return_conjs:
        return ([(c.i, c.text, c.lemma_, c.ent_type_, get_child_dict(c)) for c in doc[head_i].conjuncts]
                + head_info)
    else:
        return head_info


def get_roles_pos(tags):
    tags = np.append(tags, 'O')
    roles = []
    curr_role = None
    curr_start = None
    curr_end = None
    for i, tag in enumerate(tags):
        prefix, label = tag[0], tag[2:]
        if prefix == 'B':
            if curr_role:
                roles.append((curr_role, curr_start, curr_end))
            curr_role = label
            curr_start = i
            curr_end = i
        elif prefix == 'I':
            if curr_role and curr_role == label:
                curr_end = i
            else:
                raise InvalidIOB(i)
        elif prefix == 'O':
            if curr_role:
                roles.append((curr_role, curr_start, curr_end))
                curr_role = None
        else:
            raise InvalidIOB(i)
    return roles


def main():
    args = parse_args()

    data_loc = Path(args.data_location)

    if args.save_location:
        save_loc = Path(args.save_location)
    else:
        save_loc = data_loc / 'srl'
    save_loc.mkdir(parents=True, exist_ok=True)

    srl_verbs = pd.read_parquet(data_loc / 'srl/verbs')
    srl_toks = pd.read_parquet(data_loc / 'srl/tokens')
    srl_toks = srl_toks.merge(pd.read_parquet(data_loc / 'sentences'))

    try:
        spacy.require_gpu()
    except Exception as e:
        print(f"Failed to enable gpu: {e}")

    nlp = spacy.load('en_core_web_trf')

    tokens_info_ids = []
    tokens_info_lemmas = []
    tokens_info_poss = []
    mismatches_ids = []
    mismatches_srl = []
    mismatches_spacy = []
    failed_verbs_ids = []
    failed_verbs_verb_id = []
    failed_verbs_failure_position = []
    roles_ids = []
    roles_verb_ids = []
    roles_roles = []
    roles_c_heads = []
    roles_words = []
    roles_lemmas = []
    roles_starts = []
    roles_ends = []
    roles_ent_types = []
    roles_child_dicts = []

    extra_space_pat = re.compile(" +")

    def process_sentences(row):
        doc = nlp(extra_space_pat.sub(" ", row['sentence']))

        tokens_info_ids.append(row[args.id_fields].to_list())
        tokens_info_lemmas.append([tok.lemma_ for tok in doc])
        tokens_info_poss.append([tok.pos_ for tok in doc])

        spacy_chunk = "+-+".join([tok.text for tok in doc])
        srl_chunk = "+-+".join(row['srl_toks'])

        if spacy_chunk != srl_chunk:
            mismatches_ids.append(row[args.id_fields].to_list())
            mismatches_srl.append(srl_chunk)
            mismatches_spacy.append(spacy_chunk)
            # print(f"Mismatch found. Total: {len(mismatches_ids)}")
            return

        verbs = row[args.id_fields].to_frame().T.merge(srl_verbs)

        for verb_row in verbs.itertuples(index=False):
            id_is = [i for i, field in enumerate(verb_row._fields) if field in args.id_fields]
            try:
                roles_pos = get_roles_pos(verb_row.tags)
            except InvalidIOB as e:
                roles_pos = []
                failed_verbs_ids.append([verb_row[i] for i in id_is])
                failed_verbs_verb_id.append(verb_row.verb_id)
                failed_verbs_failure_position.append(e.position)

            for role, start, end in roles_pos:
                range_heads = get_range_head(doc, start, end, role != 'V')
                for c_head, word, lemma, ent_type, child_dict in range_heads:
                    roles_ids.append([verb_row[i] for i in id_is])
                    roles_verb_ids.append(verb_row.verb_id)
                    roles_roles.append(role)
                    roles_c_heads.append(c_head)
                    roles_words.append(word)
                    roles_lemmas.append(lemma)
                    roles_starts.append(start)
                    roles_ends.append(end)
                    roles_ent_types.append(ent_type)
                    roles_child_dicts.append(child_dict)

    tqdm.pandas(desc='Parsing and labeling SRL output sentences...')
    srl_toks.progress_apply(process_sentences, axis=1)

    pq.write_to_dataset(pa.table(list(zip(*tokens_info_ids)) + [tokens_info_lemmas, tokens_info_poss],
                                 names=args.id_fields + ['lemmas', 'poss']),
                        save_loc / 'tokens_info')
    if len(mismatches_ids) > 0:
        pq.write_to_dataset(pa.table(list(zip(*mismatches_ids)) + [mismatches_srl, mismatches_spacy],
                                     names=args.id_fields + ['srl', 'spacy']),
                            save_loc / 'parse_mismatches')
    if len(failed_verbs_ids) > 0:
        pq.write_to_dataset(pa.table(list(zip(*failed_verbs_ids)) + [failed_verbs_verb_id, failed_verbs_failure_position],
                                     names=args.id_fields + ['verb_id', 'failure_position']),
                            save_loc / 'failed_srl')
    pq.write_to_dataset(pa.table(list(zip(*roles_ids)) + [roles_verb_ids, roles_roles, roles_c_heads, roles_words, roles_lemmas,
                                                          roles_starts, roles_ends, roles_ent_types, roles_child_dicts],
                                 names=args.id_fields + ['verb_id', 'role', 'c_head', 'word', 'lemma',
                                                         'start', 'end', 'ent_type', 'child_dict']),
                        save_loc / 'roles')

    ndocs = len(srl_toks) - len(mismatches_ids)
    print(f"Found {len(roles_ids)} role{'s' if len(roles_ids) != 1 else ''} "
          f"across {ndocs} sentence{'s' if ndocs != 1 else ''} "
          f"(mean of {len(roles_ids) / ndocs if ndocs > 0 else 0:.2f} per sentence).")
    print(f"Excluded {len(mismatches_ids)} sentence{'s' if len(mismatches_ids) != 1 else ''} "
          f"({len(mismatches_ids) / len(srl_toks) if len(srl_toks) > 0 else 0:.2%}) "
          f"due to mismatch between SRL and spaCy tokenizers.")
    print(f"Excluded {len(failed_verbs_ids)} verb{'s' if len(failed_verbs_ids) != 1 else ''} "
          f"({len(failed_verbs_ids) / len(srl_verbs) if len(srl_verbs) > 0 else 0:.2%}) "
          f"due to invalid SRL parses.")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="srl_annotator",
        description="Annotates SRL parses")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset with sentence and SRL data")
    parser.add_argument('save_location', type=str, nargs='?',
                        help="folder location to save outputs (default: in the parent of data_location)")
    parser.add_argument('-id_fields', nargs='+', type=str, default=['id', 'extract_id', 'sentence_id'],
                        help="name(s) of field in parquet dataset that contains the sentence ids "
                             "(default: ['id', 'extract_id', 'sentence_id'])")

    return parser.parse_args()


if __name__ == '__main__':
    main()
