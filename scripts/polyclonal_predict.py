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

# new deps
import polyclonal

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
        "--mut-escape-df", required=True, help="polyclonal mutational effects dataframe"
    )
    parser.add_argument(
        "--activity-wt-df", required=True, help="wildtype activity dataframe"
    )
    parser.add_argument(
        "--dms-wt-seq",
        required=True,
        help="The experimental wildtype used in the dms. This must exist in the id's of the provided fasta alignment",
    )
    parser.add_argument(
        "--antibody", required=True, help="Name of the antibody used to test escape"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The experimental wildtype used in the dms. This must exist in the id's of the provided fasta alignment",
    )
    args = parser.parse_args()

    mut_escape_df = pd.read_csv(args.mut_escape_df)
    sites_to_ignore = ["214a", "214b", "214c"]
    mut_escape_df = mut_escape_df[~mut_escape_df["site"].isin(sites_to_ignore)]
    mut_escape_df["escape"] = mut_escape_df["escape_median"]

    # Instantiate a Polyclonal object with betas and wildtype activity.
    model = polyclonal.Polyclonal(
        activity_wt_df=pd.read_csv(args.activity_wt_df),
        mut_escape_df=mut_escape_df,
        data_to_fit=None,
        alphabet=polyclonal.alphabets.AAS_WITHSTOP_WITHGAP,
    )

    # Mutation calling relative to the dms wildtype sequence.
    if args.alignment[-2:] == "xz":
        with lzma.open(args.alignment, "rt") as f:
            alignment = fasta_to_df(f)
    else:
        alignment = fasta_to_df(open(args.alignment, "r"))

    dms_wildtype = alignment.loc[args.dms_wt_seq, "seq"]
    # TODO N jobs? pandarallel apply()
    alignment["aa_substitutions"] = alignment.seq.apply(
        lambda aligned_seq: mutations(dms_wildtype, aligned_seq, set(model.mutations))
    )
    alignment.reset_index(inplace=True)

    # predict probablility of escape
    concentration_dict = {
        "CC67.105": [210],
        "CC9.104": [272],
        "LyCoV-1404": [2.6],
        "NTD_5-7": [150],
    }

    escape_probs = (
        model.prob_escape(
            variants_df=alignment, concentrations=concentration_dict[args.antibody]
        )
        .drop("seq", axis=1)
        .reset_index()
    )

    ret_json = {"generated_by": {"program": "polyclonal"}, "nodes": defaultdict(dict)}
    for strain, strain_df in escape_probs.groupby("strain"):
        for idx, row in strain_df.iterrows():
            ret_json["nodes"][strain][
                f"prob_escape_{args.antibody}_c_{row.concentration}"
            ] = row.predicted_prob_escape

    write_json(ret_json, args.output)
