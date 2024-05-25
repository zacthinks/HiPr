import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re

import pyarrow as pa
import pyarrow.parquet as pq

import spacy


def main():
    args = parse_args()

    if args.save_location:
        save_loc = Path(args.save_location)
    else:
        save_loc = Path(args.data_location).parent / 'sentences'
    save_loc.mkdir(parents=True, exist_ok=True)

    colnames = args.id_fields + ([args.loc_field] if args.loc_field != '' else []) + [args.text_field]
    texts = pd.read_parquet(args.data_location, columns=colnames)
    if args.loc_field == '':
        args.loc_field = 'loc'
        texts['loc'] = 0

    if args.mask_location:
        texts = texts[pd.read_pickle(args.mask_location)]

    pattern = re.compile(args.pattern, 0 if args.case_sensitive else re.IGNORECASE)

    try:
        spacy.require_gpu()
    except Exception as e:
        print(f"Failed to enable gpu: {e}")

    nlp = spacy.load('en_core_web_sm')

    def get_sentences(text):
        doc = nlp(text)
        return [(sent.text, sent.start_char) for sent in doc.sents if pattern.search(sent.text)]

    tqdm.pandas(desc='Extracting sentences...')
    sentences = texts[args.text_field].progress_apply(get_sentences)

    table = texts[args.id_fields + [args.loc_field]].copy()
    table['sentence'] = sentences
    table = table.explode('sentence').reset_index(drop=True)
    table[['sentence', 'loc_']] = pd.DataFrame(table['sentence'].to_list())
    table['sentence_id'] = table.groupby(['id', 'extract_id']).cumcount()
    table[args.loc_field] = table[args.loc_field] + table['loc_']
    table = table.drop('loc_', axis=1)

    pq.write_to_dataset(pa.Table.from_pandas(table), save_loc)

    print(f"Found {len(table)} sentence{'s' if len(table) != 1 else ''} "
          f"across {len(texts)} document{'s' if len(texts) != 1 else ''} "
          f"(mean of {len(table) / len(texts) if len(texts) > 0 else 0:.2f} per document).")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="sentencizer",
        description="Extracts sentences using a regex expression.")
    parser.add_argument('pattern', type=str,
                        help="regex pattern to search for")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset")
    parser.add_argument('mask_location', type=str, nargs='?',
                        help="path to a .pkl file with a selection mask for the loaded data")
    parser.add_argument('save_location', type=str, nargs='?',
                        help="folder location to save outputs (default: in the parent of data_location)")
    parser.add_argument('-case_sensitive', action='store_true',
                        help='will not ignore case when searching for regex matches')
    parser.add_argument('-text_field', nargs='?', type=str, default='extract',
                        help="name of field in parquet dataset that contains the document texts (default: 'extract')")
    parser.add_argument('-id_fields', nargs='+', type=str, default=['id', 'extract_id'],
                        help="name(s) of field in parquet dataset that contains the document ids "
                             "(default: ['id', 'extract_id'])")
    parser.add_argument('-loc_field', nargs='?', type=str, default='loc',
                        help="name of field in parquet dataset that contains the document locations, "
                             "enter empty string if none exists (default: 'loc')")

    return parser.parse_args()


if __name__ == '__main__':
    main()
