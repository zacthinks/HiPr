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
        save_loc = Path(args.data_location).parent / (args.text_field + '_mask')
    save_loc.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(args.data_location)
    p_counts = table[args.text_field].str.count(r"[^.]\.[^.\d]")
    p_densities = p_counts / table[args.text_field].str.len() * 100

    if args.simple_threshold:
        mask = (p_densities <= args.simple_threshold) | (p_counts == 1)
    else:
        mins = p_densities.sort_values().unique()
        nonzero_min = mins[1] if mins[0] == 0 else mins[0]
        p_densities_nz = np.log2(p_densities[p_densities > 0])
        p_densities = np.log2(p_densities.replace(0, nonzero_min))
        table['p_densities'] = p_densities

        if args.plot_information_criteria_range:
            df = p_densities_nz.to_frame()
            n_components = np.arange(max(1, args.plot_information_criteria_range[0]), args.plot_information_criteria_range[1])
            models = [GMM(n, covariance_type='full', random_state=args.random_seed).fit(df)
                      for n in n_components]

            plt.plot(n_components, [m.bic(df) for m in models], label='BIC')
            plt.plot(n_components, [m.aic(df) for m in models], label='AIC')
            plt.legend(loc='best')
            plt.xlabel('n_components')
            plt.savefig(save_loc / "period_rate_elbow.png")

        gmm = GMM(n_components=args.n, random_state=args.random_seed, covariance_type="full", max_iter=600, tol=1e-5)
        gmm.fit(p_densities_nz.to_frame())
        table['p_group'] = gmm.predict(p_densities.to_frame())

        if args.plot_histogram:
            plt.figure()
            table.groupby('p_group').p_densities.hist(grid=True, bins=20)
            plt.savefig(save_loc / 'log_period_rate_hist_grouped.png')

        means = table.groupby('p_group').p_densities.mean()
        min_group = means[means == means.min()].index.to_list()[0]
        mask = table.p_group == min_group

    mask.to_pickle(save_loc / "mask.pkl")

    flagged_num = sum(~mask)
    print(f"Flagged {flagged_num} document{'s' if flagged_num != 1 else ''} "
          f"across {len(table)} document{'s' if len(table) != 1 else ''} "
          f"({flagged_num / len(table) if len(table) > 0 else 0:.2%}).")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="citations_detector",
        description="Creates a boolean mask for a list of texts that can be used to remove citations using period density.")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet dataset")
    parser.add_argument('save_location', nargs='?', type=str,
                        help="folder location to save outputs (default: in the parent of data_location)")
    parser.add_argument('-n', type=int, nargs='?', default=2,
                        help="number of components to look for in GMM (default: 2)")
    parser.add_argument('-text_field', '-f', nargs='?', type=str, default='extract',
                        help="name of field in parquet dataset that contains the texts (default: 'extract')")
    parser.add_argument('--plot_histogram', '-p', action='store_true',
                        help='counts number of documents that match tokens of interest without parsing')
    parser.add_argument('--plot_information_criteria_range', '-i', type=int, nargs=2,
                        help='plot the AIC and BIC for a range of potential n.')
    parser.add_argument('-random_seed', '-s', nargs='?', type=int, default=0,
                        help="seed for random sampling (default: 0)")
    parser.add_argument('--simple_threshold', '-t', nargs='?', type=float,
                        help='flags documents that exceed the specified period density threshold in percentage points. '
                             'If specified, GMM analysis will be skipped. As a rule of thumb, 3 is a good number.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
