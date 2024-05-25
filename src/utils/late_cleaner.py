import argparse

from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc


def main():
    regex_replacement_pairs = [
        ("(ca!( [a-z]+|[.,]))", "cal\\2"),
    ]

    args = parse_args()

    if args.save_location:
        save_loc = Path(args.save_location)
    else:
        save_loc = Path(args.data_location).parent / (args.text_field + '_cleaned')
    save_loc.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.data_location)
    new_col = table.column(args.text_field)
    for pair in regex_replacement_pairs:
        new_col = pc.replace_substring_regex(new_col, *pair)

    table = table.set_column(table.column_names.index(args.text_field),
                             args.text_field,
                             new_col)

    pq.write_to_dataset(table, save_loc)

    print("Cleaning complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="late_cleaner",
        description="Targeted cleaning to improve downstream analyses.")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset")
    parser.add_argument('save_location', nargs='?', type=str,
                        help="folder location to save outputs (default: in the parent of data_location)")
    parser.add_argument('-text_field', '-f', nargs='?', type=str, default='extract',
                        help="name of field in parquet dataset that contains the texts (default: 'extract')")

    return parser.parse_args()


if __name__ == '__main__':
    main()
