import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM


def main():
    args = parse_args()

    if args.save_location:
        save_loc = Path(args.save_location)
    else:
        save_loc = Path(args.data_location).parent / (args.text_field + "_mask")
    save_loc.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(args.data_location)
    p_densities = np.log2(table[args.text_field].str.count("[^.]\\.[^.]") / table[args.text_field].str.len() * 100 + .5)
    table['p_densities'] = p_densities

    if args.plot_information_criteria_range:
        df = table['p_densities'].to_frame()
        n_components = np.arange(max(1, args.plot_information_criteria_range[0]), args.plot_information_criteria_range[1])
        models = [GMM(n, covariance_type='full', random_state=args.random_seed).fit(df)
                  for n in n_components]

        plt.plot(n_components, [m.bic(df) for m in models], label='BIC')
        plt.plot(n_components, [m.aic(df) for m in models], label='AIC')
        plt.legend(loc='best')
        plt.xlabel('n_components')
        plt.savefig(save_loc / "period_rate_elbow.png")

    gmm = GMM(n_components=args.n, random_state=args.random_seed, covariance_type="full", max_iter=600, tol=1e-5)
    gmm.fit(p_densities.to_frame())
    table['p_group'] = gmm.predict(p_densities.to_frame())

    if args.plot_histogram:
        plt.figure()
        table.groupby('p_group').p_densities.hist(grid=True, bins=15)
        plt.savefig(save_loc / 'log_period_rate_hist_grouped.png')

    means = table.groupby('p_group').p_densities.mean()
    min_group = means[means == means.min()].index.to_list()[0]
    (table.p_group == min_group).to_pickle(save_loc / "mask.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="citations_detector",
        description="Creates a boolean mask for a list of texts that can be used to remove citations using period density.")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset")
    parser.add_argument('save_location', nargs='?', type=str,
                        help="folder location to save outputs (default: same as data_location)")
    parser.add_argument('n', type=int, nargs='?', default=2,
                        help="number of components to look for in GMM (default: 2)")
    parser.add_argument('-text_field', nargs='?', type=str, default='extract',
                        help="name of field in parquet dataset that contains the texts (default: 'extract')")
    parser.add_argument('--plot_histogram', '-p', action='store_true',
                        help='counts number of documents that match tokens of interest without parsing')
    parser.add_argument('--plot_information_criteria_range', '-i', type=int, nargs=2,
                        help='plot the AIC and BIC for a range of potential n.')
    parser.add_argument('-random_seed', '-s', nargs='?', type=int, default=0,
                        help="seed for random sampling (default: 0)")

    return parser.parse_args()


if __name__ == '__main__':
    main()
