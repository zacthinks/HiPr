import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import json
import pandas as pd
from tqdm import tqdm
import re
from dataclasses import dataclass

from nltk.corpus import wordnet as wn, reader as nr


@dataclass
class Role:
    role: str
    label: str
    is_wn: bool
    number: int
    idx: int = None


def role_decomposer(role, idx=None, num_pat=re.compile(r"(.*)_(\d+)$")):
    role_type, label = role.split(': ')
    result = num_pat.search(label)
    if result:
        label, number = result.group(1), int(result.group(2))
    else:
        number = 0
    is_wn = len(label.split('.')) in [3, 4]
    return Role(role_type, label, is_wn, number, idx)


def get_wn_extension(path: str):
    with open(path, "r") as fp:
        wn_e = json.load(fp)
    return wn_e


def get_namedef_pair(synset, wn_extension):
    if isinstance(synset, nr.wordnet.Synset):
        return synset.name(), synset.definition()
    elif isinstance(synset, str):
        if synset.split('.')[2] == 'x':
            x_synset = wn_extension['synsets'][synset]
            return synset, x_synset['definition']
        else:
            synset = wn.synset(synset)
            return synset.name(), synset.definition()
    else:
        raise TypeError("Either wordnet synset or string representing synset name required for synset argument")


# Unsed
def get_glosses(lemma_strs: [str], pos_str: str = None, wn_extension=None):
    if pos_str not in [wn.VERB, wn.NOUN, wn.ADJ, wn.ADV]:
        pos_str = None

    wn_synsets = []
    x_synsets = []
    for lemma_str in lemma_strs:
        synsets = [synset for synset in wn.synsets(lemma_str, pos=pos_str) if synset not in wn_synsets]
        wn_synsets += synsets

        if wn_extension:
            if lemma_str in wn_extension['blacklisted_lemmas']:
                return []
            synsets = [synset for synset in wn_extension['lemmas'].get(lemma_str, []) if synset not in x_synsets]
            if pos_str:
                synsets = [synset for synset in synsets if synset.split('.')[1] == pos_str]
            x_synsets += synsets
    return [get_namedef_pair(s, wn_extension) for s in wn_synsets + x_synsets]


def get_hypernyms(label, wn_extension=None):
    hypernyms = set()
    to_check = [label]
    while to_check:
        label = to_check.pop()
        if label[:3] == 'OOV':
            continue
        hypernyms.add(label)
        if label.split('.')[2] != 'x':
            synset = wn.synset(label)
            to_check += ([s.name() for s in synset.hypernyms()]
                         + [s.name() for s in synset.instance_hypernyms()])
        if wn_extension:
            to_check += (wn_extension['synsets']
                         .get(label, {})
                         .get('hypernyms', []))
    return hypernyms


ENT_TO_WN = {
    'PERSON': ['person.n.01'],
    'WORK_OF_ART': ['work.n.02'],
    'ORG': ['organization.n.01'],
    'DATE': ['time_unit.n.01', 'time_period.n.01'],
    'NORP': ['social_group.n.01', 'people.n.01', 'community.n.06'],
    'CARDINAL': ['number.n.02'],
    'GPE': ['district.n.01'],
    'PERCENT': ['percentage.n.01'],
    'LOC': ['location.n.01'],
    'LANGUAGE': ['language.n.01'],
    'PRODUCT': ['product.n.02'],
    'LAW': ['document.n.01'],
    'EVENT': ['event.n.01'],
    'FAC': ['facility.n.01'],
    'MONEY': ['monetary_unit.n.01'],
    'QUANTITY': ['definite_quantity.n.01'],
    'TIME': ['time_unit.n.01', 'time_period.n.01']
}


def get_hypernyms_row(row, wn_extension=None, include_lemma=True, dummy='__', dummy_exclusion_roles=['V', 'ARG0', 'ARG1', 'ARG2']):
    hypernyms = set()
    labels = [row['wn_label']] + ENT_TO_WN.get(row['ent_type'], [])
    for label in labels:
        hypernyms |= get_hypernyms(label, wn_extension)

    if include_lemma:
        hypernyms.add(row['lemma'])
    if dummy and row['role'] not in dummy_exclusion_roles:
        hypernyms.add(dummy)
    return hypernyms


def get_role(row):
    return [f"{row['role']}: {s}" for s in row['hypernyms']]


def get_roles(group):
    role_counts = {}

    def number_roles(roles):
        numbered_roles = set()
        for role in roles:
            if role not in role_counts:
                role_counts[role] = 0
                numbered_roles.add(role)
            else:
                role_counts[role] += 1
                numbered_roles.add(f"{role}_{role_counts[role]}")
        return numbered_roles

    roles_list = group.apply(get_role, axis=1)
    return roles_list.apply(number_roles)


def main():
    # Prepare parsed arguments
    args = parse_args()

    data_loc = Path(args.data_location)
    if args.save_location:
        save_loc = Path(args.save_location)
    else:
        save_loc = data_loc.parent / 'propositions'
    save_loc.mkdir(parents=True, exist_ok=True)

    if not args.vrm:
        args.all_vrm = False
        if not args.vpm:
            print("No output type indicated--use either VFM and/or VPM flags.")
            return

    # Prepare data
    roles = pd.read_parquet(data_loc)
    sent_ids = args.id_fields
    verb_ids = sent_ids + ['verb_id']
    roles['compound_id'] = roles.groupby(verb_ids).ngroup()
    pq.write_to_dataset(
        pa.Table.from_pandas(
            roles[verb_ids + ['compound_id']].drop_duplicates().sort_values('compound_id')
        ), save_loc / 'compound_ids')

    if args.wordnet_extension_location:
        wn_extension = get_wn_extension(args.wordnet_extension_location)
    else:
        wn_extension = None

    # Masking synsets
    if args.masked_synsets:
        roles.wn_label = roles.wn_label.replace(args.masked_synsets, 'OOV')

    # Filtering propositions
    if args.keyword_pattern:
        original_len = len(roles.compound_id.unique())
        compound_ids_of_interest = roles[roles.word.str.contains(args.keyword_pattern, case=args.keyword_case_sensitive)].compound_id
        roles = roles[roles.compound_id.isin(compound_ids_of_interest)].copy()
        print(f"Filtered out {1 - (len(roles.compound_id.unique()) / original_len):.2%} of verbs that didn't include roles that matched the keyword pattern.")

    # Start main processing
    # # Get numbered roles
    roles['hypernyms'] = roles.apply(get_hypernyms_row, axis=1, wn_extension=wn_extension)
    numbered_roles = (roles.groupby('compound_id')
                      .apply(get_roles)
                      .reset_index(name='numbered_roles')
                      .drop('level_1', axis=1))

    # # Create first VRM (might take a few minutes to run)
    # # # TODO: find a better way to do this, ideally with sparse matrices
    print("Creating initial VRM...")
    vrm = (pd.get_dummies(numbered_roles.explode('numbered_roles'),
                          prefix='', prefix_sep='')
           .groupby('compound_id')
           .apply(sum)
           .drop('compound_id', axis=1))
    if args.all_vrm:
        vrm.to_pickle(save_loc / 'initial_vrm.pkl')

    # # Trim VRM
    print("Trimming VRM...")
    role_counts = vrm.sum().sort_values(ascending=False)
    vrm = vrm[role_counts[role_counts > len(vrm) * args.min_role_freq].index]
    if args.all_vrm:
        vrm.to_pickle(save_loc / 'trimmed_vrm.pkl')

    # # Remove duplicative roles from VRM
    if not args.keep_duplicative_roles:
        to_remove = set()
        for i in tqdm(range(vrm.shape[1] - 1), desc="Removing duplicative roles"):
            if i in to_remove:
                continue
            to_check = {i}
            j = 1
            while (i + j < vrm.shape[1]) and (vrm.iloc[:, i].sum() == vrm.iloc[:, i + j].sum()):
                if all(vrm.iloc[:, i] == vrm.iloc[:, i + j]):
                    to_check.add(i + j)
                j += 1

            while len(to_check) > 1:
                all_roles = [role_decomposer(vrm.columns[x], x) for x in to_check]
                roles = {f_role.role for f_role in all_roles}
                for role in roles:
                    f_roles = [f_role for f_role in all_roles if f_role.role == role]
                    if len(f_roles) == 1:
                        to_check.discard(f_roles[0].idx)
                        continue
                    is_wns = [role.is_wn for role in f_roles]
                    if set(is_wns) == {True, False}:
                        non_wns = {f_role.idx for f_role in f_roles if not f_role.is_wn}
                        to_remove |= non_wns
                        to_check -= non_wns
                    else:
                        narrowest_feats = [f_roles.pop()]
                        narrowest_hypernyms = [get_hypernyms(narrowest_feats[0].label, wn_extension)]
                        while f_roles:
                            contender_feat = f_roles.pop()
                            failed_to_remove = True
                            for n_i, narrowest_feat in enumerate(narrowest_feats):
                                if contender_feat.label in narrowest_hypernyms[n_i]:
                                    to_remove.add(contender_feat.idx)
                                    to_check.discard(contender_feat.idx)
                                    failed_to_remove = False
                                    break
                                else:
                                    contender_hypernyms = get_hypernyms(contender_feat.label, wn_extension)
                                    if narrowest_feat.label in contender_hypernyms:
                                        to_remove.add(narrowest_feat.idx)
                                        to_check.discard(narrowest_feat.idx)
                                        narrowest_feats[n_i] = contender_feat
                                        narrowest_hypernyms[n_i] = contender_hypernyms
                                        failed_to_remove = False
                                        break
                            if failed_to_remove:
                                narrowest_feats.append(contender_feat)
                                narrowest_hypernyms.append(contender_hypernyms)
                                to_check.discard(contender_feat.idx)

        vrm = vrm.drop(columns=vrm.columns[list(to_remove)], axis=1)

    if args.vrm:
        vrm.to_pickle(save_loc / 'no_duplicative_vrm.pkl')
        if not args.vpm:
            return

    # # Filtering numbered roles
    numbered_roles['trimmed_roles'] = numbered_roles.numbered_roles.apply(lambda feats: feats & set(vrm.columns))
    numbered_roles['trimmed_n'] = numbered_roles['trimmed_roles'].apply(len)
    print(f"{sum(numbered_roles.trimmed_n == 0) / len(numbered_roles):.2%} of {len(numbered_roles)} roles trimmed.")
    numbered_roles = numbered_roles[numbered_roles.trimmed_n > 0].copy()

    # # Recover propositions that meet some frequency threshold
    min_count = len(vrm) * args.min_prop_freq

    # # # Get keyword (or starting) roles
    if args.keyword_pattern:
        keyword_pat = re.compile(args.keyword_pattern, flags=re.IGNORECASE if not args.keyword_case_sensitive else 0)
        keyword_roles = [role for role in vrm.columns.to_list()
                         if keyword_pat.match(role.split(': ')[1])]
    else:
        role_counts = vrm.sum()
        keyword_roles = role_counts[role_counts > min_count].index.to_list()

    # # # Iteratively recover propositions that meet the minimum count
    propositions_list = [keyword_roles]
    proposition_counts_list = [vrm[keyword_roles].sum()]

    n = 1
    while True:
        if n == args.max_proposition_n:
            break
        propositions = []
        proposition_counts = []
        for prop in tqdm(propositions_list[-1], desc=f"Looking for propositions of length n={n+1}"):
            prev_roles = set(prop.split('; '))
            numbered_roles['flagged'] = numbered_roles.trimmed_roles.apply(lambda s: len(s & prev_roles) > 0)
            matched_props = numbered_roles.groupby('compound_id').flagged.sum() == n
            matched_props = matched_props[matched_props].index
            numbered_roles_filtered = numbered_roles[numbered_roles.compound_id.isin(matched_props)]
            numbered_roles_filtered = numbered_roles_filtered[~numbered_roles_filtered.flagged]
            next_roles = numbered_roles_filtered.trimmed_roles.explode().value_counts()
            next_roles = next_roles[next_roles >= min_count]
            next_props = next_roles.index.map(lambda next_role: '; '.join(sorted(prev_roles | {next_role})))
            mask = [next_prop not in propositions for next_prop in next_props]
            propositions += next_props[mask].to_list()
            proposition_counts += next_roles[mask].to_list()
        if propositions:
            propositions_list.append(propositions)
            proposition_counts_list.append(proposition_counts)
            n += 1
        else:
            break

    # # # Clean up dataframes
    print("Cleaning up dataframes...")

    def get_n_props_df(n):
        n_props_df = pd.DataFrame({'prop': propositions_list[n], 'count': proposition_counts_list[n]})
        n_props_df = n_props_df.sort_values('count', ascending=False)
        n_props_df['doc_prop'] = n_props_df['count'] / len(vrm)
        return n_props_df

    def props_df_cleaner(i, df):
        df['n'] = i + 1
        df.reset_index(drop=True, inplace=True)
        return df

    n_props_dfs = [get_n_props_df(n) for n in range(len(propositions_list))]
    all_props = pd.concat([props_df_cleaner(i, df) for i, df in enumerate(n_props_dfs)]).reset_index(drop=True)

    if args.prop_counts:
        all_props.to_pickle(save_loc / 'all_prop_counts.pkl')

    # # Construct VPM
    print("Constructing VPM...")

    def get_prop_verbs(prop):
        roles = prop.split('; ')
        return vrm[roles].sum(axis=1) == len(roles)

    vpm = pd.DataFrame(zip(*[get_prop_verbs(prop) for prop in all_props.prop.to_list()]), columns=all_props.prop.to_list())
    vpm.index = vrm.index
    vpm.to_pickle(save_loc / 'vpm.pkl')
    print(f"VPM created with {quantify_noun(len(all_props), 'proposition')}.")


def quantify_noun(quantity, noun, format='', irregular=None):
    if irregular:
        sin, plu = irregular
    else:
        sin, plu = '', 's'
    return f"{quantity:{format}} {noun}{plu if quantity != 1 else sin}"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="proposition identifier",
        description="Creates propositional representations of data, i.e., verb feature matrices (verb role "
                    "matrices or verb proposition matrices) and proposition counts, from annotated semantic roles.")
    parser.add_argument('data_location', type=str,
                        help="folder containing parquet datasets with roles_wn data")
    parser.add_argument('-s', '--save_location', type=str,
                        help="folder location to save outputs (default: in the parent of data_location)")
    parser.add_argument('-r', '--vrm', action='store_true',
                        help='save verb role matrix (VRM) objects')
    parser.add_argument('-a', '--all_vrm', action='store_true',
                        help='save all VRM objects including intermediate ones')
    parser.add_argument('-c', '--prop_counts', action='store_true',
                        help='save proposition counts (a by-product of creating the VPM so only relevant if VPM is turned on)')
    parser.add_argument('-p', '--vpm', action='store_true',
                        help='create and save verb proposition matrix (VPM) objects '
                             '(creating VPMs can take a while--increase min_role_freq or min_prop_freq to speed up)')
    parser.add_argument('-n', '--max_proposition_n', type=int, default=-1,
                        help='cap the number of roles when looking for propositions (will speed things up)')
    parser.add_argument('-d', '--keep_duplicative_roles', action='store_true',
                        help='by default, roles that are deemed duplicative are removed--use this flag to keep all roles')
    parser.add_argument('-k', '--keyword_pattern', type=str,
                        help="regex pattern for keyword(s) all propositions should include "
                             "(including this greatly speeds up the process of finding propositions)")
    parser.add_argument('--keyword_case_sensitive', '--case', action='store_true',
                        help="make the keyword pattern match case sensitive")
    parser.add_argument('-i', '--id_fields', nargs='+', type=str, default=['id', 'extract_id', 'sentence_id'],
                        help="name(s) of field in parquet dataset that contains the sentence ids "
                             "(default: ['id', 'extract_id', 'sentence_id'])")
    parser.add_argument('-m', '--masked_synsets', nargs='+', type=str,
                        help="wordnet synsets to mask")
    parser.add_argument('-x', '--wordnet_extension_location', type=str,
                        help="location of wordnet extensions")
    parser.add_argument('--min_role_freq', '--min_r', type=float, default=.003,
                        help="minimum number of verbs that need to have a role, otherwise it will be trimmed "
                             "(values less than 1 will be treated as proportions of total verbs, default: .003)")
    parser.add_argument('--min_prop_freq', '--min_p', type=float, default=.005,
                        help="minimum number of verbs that need to have a proposition, otherwise it will be trimmed "
                             "(values less than 1 will be treated as proportions of total verbs, default: .005)")
    return parser.parse_args()


if __name__ == '__main__':
    main()
