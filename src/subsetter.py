import argparse
from glob import glob
import numpy as np
from tqdm import tqdm
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


def interval_merger(raws):
    intervals = [raws[0]]
    for raw in raws[1:]:
        if intervals[-1][1] > raw[0]:
            intervals[-1] = (intervals[-1][0], raw[1])
        else:
            intervals.append(raw)
    return intervals


def main():
    args = parse_args()

    np.random.seed(args.random_seed)
    pattern = re.compile(args.pattern, 0 if args.case_sensitive else re.IGNORECASE)

    def get_windows(text):
        raws = [(max(0, match.start() - args.window), match.end() + args.window)
                for match in pattern.finditer(text)]
        intervals = [raws[0]]
        for raw in raws[1:]:
            if intervals[-1][1] > raw[0]:
                intervals[-1] = (intervals[-1][0], raw[1])
            else:
                intervals.append(raw)

        return [text[interval[0]:interval[1]] for interval in intervals]

    data_locs = glob(args.data_location + "/*.parquet")

    total_docs = 0
    total_matches = 0
    for data_loc in tqdm(data_locs, desc='parquet files'):
        table = pq.read_table(data_loc)
        matches = pc.count_substring_regex(
            table.column(args.text_field),
            args.pattern,
            ignore_case=not args.case_sensitive)
        total_matches += pc.sum(matches).as_py()
        mask = pc.greater(matches, 0)
        total_docs += pc.sum(mask).as_py()

        if args.count_only:
            continue

        table = table.filter(mask).to_pandas()
        table['extract'] = table[args.text_field].apply(get_windows)
        if args.text_field != 'extract':
            table = table.drop(columns=args.text_field)
        table = table.explode('extract')
        table['extract_id'] = table.groupby(args.id_field).cumcount()
        table = pa.Table.from_pandas(table, preserve_index=False)
        pq.write_to_dataset(table, args.save_location + '/extracts')

    print(f"Found {total_matches} matche{'s' if total_matches != 1 else ''} "
          f"across {total_docs} document{'s' if total_docs != 1 else ''} "
          f"(mean of {total_matches / total_docs if total_docs > 0 else 0:.2f} per document).")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="subsetter",
        description="Creates a limited dataset using a regex expression.")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset     ")
    parser.add_argument('save_location', type=str,
                        help="folder location to save outputs")
    parser.add_argument('pattern', type=str,
                        help="regex pattern to search for")
    parser.add_argument('-case_sensitive', action='store_true',
                        help='will not ignore case when searching for regex matches')
    parser.add_argument('-text_field', nargs='?', type=str, default='fulltext',
                        help="name of field in parquet dataset that contains the document texts (default: 'fulltext')")
    parser.add_argument('-id_field', nargs='?', type=str, default='id',
                        help="name of field in parquet dataset that contains the document ids (default: 'id')")
    parser.add_argument('-window', nargs='?', type=int, default=500,
                        help="size of search window before and after token (default: 500)")
    parser.add_argument('-count_only', action='store_true',
                        help='counts number of documents that match tokens of interest without parsing')
    parser.add_argument('-random_sample', '-r', nargs='?', type=float, default=1,
                        help="probability of including document in sample (default: 1)")
    parser.add_argument('-random_seed', '-s', nargs='?', type=int, default=0,
                        help="seed for random sampling (default: 0)")

    return parser.parse_args()


if __name__ == '__main__':
    main()
