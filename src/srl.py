import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from allennlp_models import pretrained


def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    if args.save_location:
        save_loc = Path(args.save_location)
    else:
        save_loc = Path(args.data_location).parent / 'srl'
    save_loc.mkdir(parents=True, exist_ok=True)

    colnames = args.id_fields + [args.text_field]
    texts = pd.read_parquet(args.data_location, columns=colnames).sample(frac=args.random_sample)

    if args.mask_location:
        texts = texts[pd.read_pickle(args.mask_location)]

    try:
        predictor = pretrained.load_predictor("structured-prediction-srl-bert", cuda_device=0)
    except Exception as e:
        print(f"Failed to load AllenNLP pretrained model with GPU: {e}")
        print(f"Trying without GPU...")
        try:
            predictor = pretrained.load_predictor("structured-prediction-srl-bert", cuda_device=-1)
        except Exception as e:
            print(f"Failed tp load model: {e}")
            return

    def srl_info(row):
        try:
            srl = predictor.predict(row[args.text_field])
            row['srl_toks'] = srl['words']
            row['srl_verbs'] = srl['verbs']
        except Exception as e:
            print(f'Failed to parse ({e}): {row[args.text_field]}')
            row['srl_toks'] = []
            row['srl_verbs'] = []

        return row

    tqdm.pandas(desc='Labeling sentences...')
    texts = texts.progress_apply(srl_info, axis=1).drop(args.text_field, axis=1)

    # texts.to_pickle(save_loc / "srl.pkl")

    table = texts[args.id_fields].copy()
    table['verb_dict'] = texts.pop("srl_verbs")
    table = table.explode('verb_dict').reset_index(drop=True)
    table['verb_id'] = table.groupby(args.id_fields).cumcount()

    def verb_expander(row):
        d = row['verb_dict']
        if isinstance(d, dict):
            row['tags'] = d['tags']
        else:
            row['tags'] = []
        return row

    table = table.apply(verb_expander, axis=1)
    table = table.drop('verb_dict', axis=1)

    pq.write_to_dataset(pa.Table.from_pandas(texts), save_loc / 'tokens')
    pq.write_to_dataset(pa.Table.from_pandas(table), save_loc / 'verbs')

    print(f"Found {len(table)} verb{'s' if len(table) != 1 else ''} "
          f"across {len(texts)} document{'s' if len(texts) != 1 else ''} "
          f"(mean of {len(table) / len(texts) if len(texts) > 0 else 0:.2f} per document).")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="srl",
        description="Extracts SRL tokens and verb dictionaries.")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset")
    parser.add_argument('mask_location', type=str, nargs='?',
                        help="path to a .pkl file with a selection mask for the loaded data")
    parser.add_argument('save_location', type=str, nargs='?',
                        help="folder location to save outputs (default: in the parent of data_location)")
    parser.add_argument('-text_field', nargs='?', type=str, default='sentence',
                        help="name of field in parquet dataset that contains the sentences (default: 'sentence')")
    parser.add_argument('-id_fields', nargs='+', type=str, default=['id', 'extract_id', 'sentence_id'],
                        help="name(s) of field in parquet dataset that contains the document ids "
                             "(default: ['id', 'extract_id', 'sentence_id'])")
    parser.add_argument('-random_sample', '-r', nargs='?', type=float, default=1,
                        help="probability of including document in sample (default: 1)")
    parser.add_argument('-random_seed', '-s', nargs='?', type=int, default=0,
                        help="seed for random sampling (default: 0)")

    return parser.parse_args()


if __name__ == '__main__':
    main()
