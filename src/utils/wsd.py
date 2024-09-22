import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import json
import pandas as pd
from tqdm import tqdm
import re

import src.scripts.model.wordnet_consec as consec


def main():
    args = parse_args()

    mt = consec.get_consec_module_tokenizer(args.consec_location,
                                            args.cuda_device)
    if args.wordnet_extension_location:
        wn_extension = consec.get_wn_extension(args.wordnet_extension_location)
    else:
        wn_extension = None

    data_loc = Path(args.data_location)
    if args.save_location:
        save_loc = Path(args.save_location)
    else:
        save_loc = data_loc
    save_loc.mkdir(parents=True, exist_ok=True)

    tokens = pd.read_parquet(data_loc / 'tokens').merge(pd.read_parquet(data_loc / 'tokens_info'))
    roles = pd.read_parquet(data_loc / 'roles')

    if not args.keep_rc:
        roles = roles[roles.role.str[0].isin(['A', 'V'])]
    if args.verb_filter_pattern:
        filtered_ids = (roles[roles.word.str.contains(args.verb_filter_pattern, flags=re.IGNORECASE)]
                        [args.id_fields + ['verb_id']]
                        .drop_duplicates())
        roles = filtered_ids.merge(roles)
    roles = roles.merge(tokens)

    definitions = {}
    oov_count = 0

    def head_wsd(row):
        nonlocal oov_count

        forced_pos = 'v' if row['role'] == 'V' else None
        label, d, conf = consec.wordnet_consec(mt, target_position=row['c_head'], tokens=row['srl_toks'],
                                               poss=row['poss'], forced_pos=forced_pos, lemmas=row['lemmas'],
                                               use_lemmas=True, use_both=True,
                                               wn_extension=wn_extension, full_consec=False)
        # todo: find way to include full consec--currently including all context definitions results in exceeding token limit (1024)

        if label[:3] == 'OOV':
            oov_count += 1
        else:
            if label not in definitions:
                definitions[label] = d

        row['wn_label'] = label
        row['wn_confidence'] = conf

        return row

    tqdm.pandas(desc='Disambiguating tokens')
    wsd_results = (roles
                   .progress_apply(head_wsd, axis=1)
                   .drop(columns=['srl_toks', 'lemmas', 'poss']))

    with open(save_loc / 'wn_dictionary.json', 'w') as out:
        json.dump(definitions, out)

    pq.write_to_dataset(pa.Table.from_pandas(wsd_results), save_loc / 'roles_wn')

    disambiguated_count = len(wsd_results) - oov_count
    print(f"Disambiguated {disambiguated_count} role{'s' if disambiguated_count != 1 else ''} "
          f"({disambiguated_count / len(wsd_results) if len(wsd_results) > 0 else 0:.2%}) "
          f"with a total of {len(definitions)} unique synset{'s' if len(definitions) != 1 else ''}.")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="wordnet_wsd",
        description="Disambiguates tokens using wordnet and consec.")
    parser.add_argument('consec_location', type=str,
                        help="location of consec checkpoint")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset with SRL data (tokens, tokens_info, roles)")
    parser.add_argument('save_location', type=str, nargs='?',
                        help="folder location to save outputs (default: same as data_location)")
    parser.add_argument('--wordnet_extension_location', '-x', type=str, nargs='?',
                        help="location wordnet extensions)")
    parser.add_argument('--verb_filter_pattern', '-f', type=str, nargs='?',
                        help="regex pattern used to filter verbs to only those that contain at least one role that matches the pattern")
    parser.add_argument('--keep_rc', '-k', action='store_true',
                        help="keep R- and C- roles, which are discarded by default")
    parser.add_argument('--id_fields', '-i', nargs='+', type=str, default=['id', 'extract_id', 'sentence_id'],
                        help="name(s) of field in parquet dataset that contains the sentence ids "
                             "(default: ['id', 'extract_id', 'sentence_id'])")
    parser.add_argument('--cuda_device', '-c', nargs='?', type=int, default=0,
                        help="cuda device to use, -1 for CPU (default: 0)")

    return parser.parse_args()


if __name__ == '__main__':
    main()
