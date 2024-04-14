import os
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
import gzip
import json
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import json as pjson
import pyarrow.compute as pc

import fasttext

from utils.page_merger import join_pages
from utils.lang_checker import lang_checker
from utils.cleaner import cleaner
# include doc lengths


def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    # Check for save location
    save_loc = Path(args.save_location)
    save_loc.mkdir(parents=True, exist_ok=True)

    # Find jsonl.gz files
    data_locs = glob(args.data_location)
    if len(data_locs) > 0:
        print(f"{len(data_locs)} jsonl.gz files found.")
    else:
        print(f"No jsonl.gz files found.")
        return

    # Load language detection model
    if os.path.exists('src/utils/lid.176.bin'):
        model = fasttext.load_model('src/utils/lid.176.bin')
    else:
        print("FastText pretrained model for language identification not found in utils folder.")
        return

    # Load list of journals to filter by
    if args.journals:
        journals_tbl = pq.read_table(args.journals)

    ids = {' '}
    # Loop through each each jsonl.gz file
    for data_loc in tqdm(data_locs, desc='jsonl.gz files'):
        if args.line:
            original_len = 0
            count = 0
            metadata_list = []
            id_list = []
            fulltext_list = []
            with gzip.open(data_loc, 'rt') as f:
                for line in tqdm(f, desc='Reading lines', total=100000):
                    if np.random.rand() > args.random_sample:
                        continue
                    original_len += 1
                    data = json.loads(line)
                    if not pc.is_in(data['isPartOf'], journals_tbl['journal']).as_py():
                        continue
                    data['id'] = data['id'][28:]
                    if data['id'] in ids:
                        continue
                    fulltext = data.pop('fullText')
                    lang_data = lang_checker(fulltext, model, args.window, args.number).to_dict()
                    data.update(lang_data)
                    metadata_list.append(data)
                    if (data['top_lang'] == args.language
                            and data['lang_score_' + args.measure] * data['top_lang_prop'] >= args.threshold):
                        count += 1
                        id_list.append(data['id'])
                        fulltext = join_pages(fulltext)
                        fulltext_list.append(cleaner(fulltext))
                    if len(metadata_list) == args.batch_size:
                        table = pa.Table.from_pylist(metadata_list)
                        pq.write_to_dataset(table, save_loc / 'metadata')
                        table = pa.Table.from_pydict({
                            'id': id_list,
                            'fulltext': fulltext_list})
                        pq.write_to_dataset(table, save_loc / 'fulltext')
                        metadata_list = []
                        id_list = []
                        fulltext_list = []
                if len(metadata_list) > 0:
                    table = pa.Table.from_pylist(metadata_list)
                    pq.write_to_dataset(table, save_loc / 'metadata')
                    table = pa.Table.from_pydict({
                        'id': id_list,
                        'fulltext': fulltext_list})
                    pq.write_to_dataset(table, save_loc / 'fulltext')

            print(f"Saved {count} documents out of {original_len} read documents from {data_loc}.")

        else:
            table = pjson.read_json(data_loc, pjson.ReadOptions(block_size=10 << 20))
            original_len = len(table)
            curr_len = original_len
            table = table.set_column(
                table.column_names.index('id'),
                'id',
                pc.ascii_ltrim(table['id'], "http://www.jstor.org/stable/"))

            # get random subset
            if args.random_sample != 1:
                table = table.filter(np.random.rand(original_len) <= args.random_sample)
                curr_len = len(table)
                tqdm.write(f"Random sample (seed: {args.random_seed}) of {curr_len} document"
                           f"{'s' if curr_len != 1 else ''} ({curr_len / original_len:.2%}) taken"
                           f"from an initial {original_len} document{'s' if original_len != 1 else ''}")
                original_len = curr_len

            # remove unwanted journals
            if args.journals:
                mask = pc.is_in(table['isPartOf'], journals_tbl['journal'])
                table = table.filter(mask)
                num_removed = curr_len - len(table)
                curr_len = len(table)
                tqdm.write(f"{num_removed} document{'s' if num_removed != 1 else ''} "
                           f"({num_removed / original_len:.2%}) outside of specified journal list removed.")

            # remove duplicates
            mask = pc.invert(pc.is_in(table['id'], pa.array(ids)))
            table = table.filter(mask)
            num_removed = curr_len - len(table)
            curr_len = len(table)
            tqdm.write(f"{num_removed} duplicate document{'s' if num_removed != 1 else ''} "
                       f"({num_removed / original_len:.2%}) removed.")

            # convert fulltexts into a pandas series to apply wrangling
            fulltexts = table['fullText'].to_pandas()

            tqdm.pandas(desc='Checking languages...')
            langs_df = fulltexts.progress_apply(lambda x: lang_checker(x, model, args.window, args.number))
            table = table.append_column('top_lang', pa.array(langs_df.top_lang))
            table = table.append_column('lang_score_max', pa.array(langs_df.lang_score_max))
            table = table.append_column('lang_score_mean', pa.array(langs_df.lang_score_mean))
            table = table.append_column('lang_score_median', pa.array(langs_df.lang_score_median))
            table = table.append_column('top_lang_prop', pa.array(langs_df.top_lang_prop))

            # save metadata to file along with new language data
            colnames = table.column_names
            colnames.remove('fullText')
            pq.write_to_dataset(table.select(colnames), save_loc / 'metadata')

            # restrict table and fulltexts to appropriate language, update ids set
            mask_lang = pc.equal(table['top_lang'], args.language)
            mask_val = pc.greater_equal(pc.multiply(table['lang_score_' + args.measure], table['top_lang_prop']),
                                        args.threshold)
            mask = pc.and_(mask_lang, mask_val)
            table = table.filter(mask)
            fulltexts = fulltexts[mask.to_pandas()]
            ids.update(table['id'].to_pylist())
            num_removed = curr_len - len(table)
            curr_len = len(table)
            tqdm.write(f"{num_removed} document{'s' if num_removed != 1 else ''} ({num_removed / original_len:.2%}) "
                       "not matching the specified language requirements removed.")

            tqdm.pandas(desc='Merging pages...')
            fulltexts = fulltexts.progress_apply(join_pages)

            tqdm.pandas(desc='Cleaning documents...')
            fulltexts = fulltexts.progress_apply(cleaner)

            # save cleaned docs with ids
            table = pa.Table.from_pydict({
                'id': table['id'],
                'fulltext': fulltexts})
            pq.write_to_dataset(table, save_loc / 'fulltext')

            print(f"Saved {len(table)} documents out of {original_len} read documents from {data_loc}.")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="cleaner",
        description="Parses Constellate jsonl.gz files to extract and clean full text data and metadata.")
    parser.add_argument('data_location', type=str,
                        help="location of Constellate jsonl.gz file(s) in terms of glob pattern")
    parser.add_argument('save_location', type=str,
                        help="folder location to save outputs")
    parser.add_argument('-line', action='store_true',
                        help="read jsonl.gz files line by line instead of as an entire table, helpful if RAM is low. "
                             "Note that this is significantly slower, not just because each row as to be read but also "
                             "because computations have to be done row by row as well.")
    parser.add_argument('-window', nargs='?', type=int, default=1000,
                        help="size of each sample window of text to be extracted for language checks (default: 1000)")
    parser.add_argument('-number', nargs='?', type=int, default=10,
                        help="number of sample windows to extract for language checks (default: 10)")
    parser.add_argument('-measure', nargs='?', type=str, default='median',
                        help="measure of central tendency to use for checking if a document meets the minimum threshold "
                             "in the specified language for inclusion in cleaned dataset (options: 'median', 'mean', "
                             "'max', default: 'median')")
    parser.add_argument('-language', nargs='?', type=str, default='en',
                        help="language of documents to include in cleaned dataset (default: 'en')")
    parser.add_argument('-journals', type=str,
                        help="table (parquet) of journals to restrict dataset to")
    parser.add_argument('-threshold', nargs='?', type=float, default=.8,
                        help="language score threshold for including a document in cleaned dataset (default: .8)")
    parser.add_argument('-random_sample', '-r', nargs='?', type=float, default=1,
                        help="probability of including document in sample (default: 1)")
    parser.add_argument('-random_seed', '-s', nargs='?', type=int, default=0,
                        help="seed for random sampling (default: 0)")
    parser.add_argument('-batch_size', nargs='?', type=int, default=10000,
                        help="maximum number of documents per file in cleaned dataset, only applies if data is processed "
                             "line by line (default: 10000)")

    return parser.parse_args()


if __name__ == '__main__':
    main()
