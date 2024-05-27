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


def get_range_head(doc, start, end, return_conjs):
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

    if return_conjs:
        return ([(c.i, c.ent_type_, c.pos_, get_child_dict(c)) for c in doc[head_i].conjuncts]
                + [(head_i, doc[head_i].ent_type_, doc[head_i].pos_, get_child_dict(doc[head_i]))])
    else:
        return [(head_i, doc[head_i].ent_type_, doc[head_i].pos_, get_child_dict(doc[head_i]))]


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

    mismatches = pd.DataFrame(columns=args.id_fields + ['srl', 'spacy'])
    failed_verbs = pd.DataFrame(columns=args.id_fields + ['verb_id', 'failure_position'])
    srl_roles = pd.DataFrame(columns=(args.id_fields + ['verb_id',
                                                        'role', 'c_head', 'start', 'end',
                                                        'ent_type', 'pos', 'word', 'lemma', 'child_dict']))
    extra_space_pat = re.compile(" +")

    def process_sentences(row):
        doc = nlp(extra_space_pat.sub(" ", row['sentence']))
        #spacy_indices = [tok.i for tok in doc if tok.text.strip() != ""]
        #spacy_chunk = "+-+".join([doc[i].text for i in spacy_indices])
        spacy_chunk = "+-+".join([tok.text for tok in doc])
        srl_chunk = "+-+".join(row['srl_toks'])

        if spacy_chunk != srl_chunk:
            mismatch_row = row[args.id_fields]
            mismatch_row['srl'] = srl_chunk
            mismatch_row['spacy'] = spacy_chunk
            mismatches.loc[len(mismatches)] = mismatch_row
            print(f"Mismatch found. Total: {len(mismatches)}")
            return

        verbs = row[args.id_fields].to_frame().T.merge(srl_verbs)

        for i, row in verbs.iterrows():
            try:
                roles_pos = get_roles_pos(row['tags'])
            except InvalidIOB as e:
                roles_pos = []
                failed_row = row[args.id_fields]
                failed_row['verb_id'] = row['verb_id']
                failed_row['failure_position'] = e.position
                failed_verbs.loc[len(failed_verbs)] = failed_row

            for role, start, end in roles_pos:
                range_heads = get_range_head(
                    doc, start, end, role != 'V')
                for c_head, ent_type, pos, child_dict in range_heads:
                    role_row = row[args.id_fields]
                    role_row['verb_id'] = row['verb_id']
                    role_row['role'] = role
                    try:
                        role_row['c_head'] = c_head
                    except ValueError:
                        print("ValueError")
                        print(start, end, c_head, pos, doc[c_head].text, doc)
                    role_row['start'] = start
                    role_row['end'] = end
                    role_row['ent_type'] = ent_type
                    role_row['pos'] = pos
                    role_row['word'] = doc[c_head].text
                    role_row['lemma'] = doc[c_head].lemma_
                    role_row['child_dict'] = child_dict

                    srl_roles.loc[len(srl_roles)] = role_row

    tqdm.pandas(desc='Parsing and labeling SRL output sentences...')
    srl_toks.progress_apply(process_sentences, axis=1)

    pq.write_to_dataset(pa.Table.from_pandas(srl_roles), save_loc / 'roles')
    pq.write_to_dataset(pa.Table.from_pandas(mismatches), save_loc / 'parse_mismatches')
    pq.write_to_dataset(pa.Table.from_pandas(failed_verbs), save_loc / 'failed_srl')

    ndocs = len(srl_toks) - len(mismatches)
    print(f"Found {len(srl_roles)} role{'s' if len(srl_roles) != 1 else ''} "
          f"across {ndocs} sentence{'s' if ndocs != 1 else ''} "
          f"(mean of {len(srl_roles) / ndocs if ndocs > 0 else 0:.2f} per sentence).")
    print(f"Excluded {len(mismatches)} sentence{'s' if len(mismatches) != 1 else ''} "
          f"({len(mismatches) / len(srl_toks) if len(srl_toks) > 0 else 0:.2%}) "
          f"due to mismatch between SRL and spaCy tokenizers.")
    print(f"Excluded {len(failed_verbs)} verb{'s' if len(failed_verbs) != 1 else ''} "
          f"({len(failed_verbs) / len(srl_verbs) if len(srl_verbs) > 0 else 0:.2%}) "
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
