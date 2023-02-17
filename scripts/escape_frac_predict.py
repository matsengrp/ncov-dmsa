#!/usr/bin/env python
"""
convert peptide metadata to fasta format.
"""

import sys
import lzma
from collections import defaultdict
import json
import argparse

# existing deps
import pandas as pd
from augur.utils import write_json
from Bio import SeqIO
from Bio.Seq import Seq


# TODO Bio.Align may be faster?
def fasta_to_df(fasta_file):
    """simply convert a fasta to dataframe"""

    ids, seqs = [], []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
        ids.append(seq_record.id)
        seqs.append(str(seq_record.seq))
    return pd.DataFrame({"strain": ids, "seq": seqs}).set_index("strain")


def mutations(naive_aa, aa, allowed_subs):
    """Amino acid substitutions between two sequences, in IMGT coordinates."""

    assert len(naive_aa) == len(aa)
    return " ".join(
        [
            f"{aa1}{pos+1}{aa2}"
            for pos, (aa1, aa2) in enumerate(zip(naive_aa, aa))
            if aa1 != aa2 and f"{aa1}{pos+1}{aa2}" in allowed_subs
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the probability of escape given ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alignment", required=True, help="fasta with protein alignments"
    )
    parser.add_argument(
        "--mut-effects-df", required=True, help="a csv with columns aa_subs ... "
    )
    parser.add_argument(
        "--dms-wt-seq",
        required=True,
        help="The experimental wildtype used in the dms. This must exist in the id's of the provided fasta alignment",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The experimental wildtype used in the dms. This must exist in the id's of the provided fasta alignment",
    )
    args = parser.parse_args()

    mut_effects_df = pd.read_csv(args.mut_effects_df)
    mut_effects_df = mut_effects_df.assign(
        non_escape_frac=(1-mut_effects_df["mut_escape_frac_epistasis_model"])
    )

    # Mutation calling relative to the dms wildtype sequence.
    if args.alignment[-2:] == "xz":
        with lzma.open(args.alignment, "rt") as f:
            alignment = fasta_to_df(f)
    else:
        alignment = fasta_to_df(open(args.alignment, "r"))

    # TODO does this really even need to be in nextstrain tree?
    dms_wildtype = alignment.loc[args.dms_wt_seq, "seq"]
    
    # TODO N jobs? pandarallel apply()
    alignment["aa_substitutions"] = alignment.seq.apply(
        lambda aligned_seq: mutations(dms_wildtype, aligned_seq, set(mut_effects_df.aa_substitution))
    )
    alignment.reset_index(inplace=True)

    def compute_variant_escape_score(
        aa_subs,
        mut_effect_col = "non_escape_frac"
    ):
        data = mut_effects_df[mut_effects_df['aa_substitution'].isin(aa_subs.split())]
        return 1-data[mut_effect_col].prod()

    alignment["variant_escape_score"] = alignment.aa_substitutions.apply(
        lambda aa_subs: compute_variant_escape_score(aa_subs)
    )

    ret_json = {"generated_by": {"program": "custom"}, "nodes": defaultdict(dict)}
    for idx, row in alignment.iterrows():
        ret_json["nodes"][row.strain][
            f"variant_escape_score"
        ] = row.variant_escape_score

    write_json(ret_json, args.output)
