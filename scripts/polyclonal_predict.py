#!/usr/bin/env python
"""
convert peptide metadata to fasta format.
"""

import pandas as pd
import sys
import lzma
import json
import argparse
import polyclonal 
from augur.utils import write_json
from Bio import SeqIO
from Bio.Seq import Seq


def fasta_to_df(fasta_file):
    """simply convert a fasta to dataframe"""

    ids, seqs = [], []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
        ids.append(seq_record.id)
        seqs.append(str(seq_record.seq))
    return pd.DataFrame({"strain":ids, "seq":seqs}).set_index("strain")

def mutations(naive_aa, aa, allowed_subs):
    """Amino acid substitutions between two sequences, in IMGT coordinates."""

    assert len(naive_aa) == len(aa)
    return " ".join([
        f"{aa1}{pos+1}{aa2}"
        for pos, (aa1, aa2) in enumerate(zip(naive_aa, aa))
        if aa1 != aa2 and f"{aa1}{pos+1}{aa2}" in allowed_subs
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the probability of escape given ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # TODO we could read in a tree and use augur api to traverse and make predictions on inferred nodes?
    # otherwise, the best option is probably to impliment a new augur command to pred on dms biophysical models 
    parser.add_argument("--alignment", required=True, help="fasta with protein alignments")
    parser.add_argument("--mut-escape-df", required=True, help="polyclonal mutational effects dataframe")
    parser.add_argument("--activity-wt-df", required=True, help="wildtype activity dataframe")
    parser.add_argument("--dms-wt-seq", required=True, help="The experimental wildtype used in the dms. This must exist in the id's of the provided fasta alignment")
    parser.add_argument("--dms-wt-seq", required=True, help="Name of the antibody used to test escape")
    parser.add_argument("--output", required=True, help="The experimental wildtype used in the dms. This must exist in the id's of the provided fasta alignment")
    args = parser.parse_args()

    # Instantiate a Polyclonal object with betas and wildtype activity.
    model = polyclonal.Polyclonal(
        activity_wt_df=pd.read_csv(args.activity_wt_df),
        mut_escape_df=pd.read_csv(args.mut_escape_df),
        data_to_fit=None,
        alphabet=polyclonal.alphabets.AAS_WITHSTOP_WITHGAP
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
        lambda aligned_seq: mutations(
            dms_wildtype, 
            aligned_seq, 
            set(model.mutations)
        )
    )
    alignment.reset_index(inplace=True)

    # predict probablility of escape
    escape_probs = model.prob_escape(
        variants_df=alignment,
        concentrations=[2.6]
    )

    ret_json = {"generated_by":{"program":"polyclonal"}, "nodes":{}}
    for idx, row in escape_probs.drop("seq", axis=1).reset_index().iterrows():
        ret_json["nodes"][row.strain] = {f"prob_escape_{args.antibody}" : row.predicted_prob_escape}

    #j = json.dumps(ret_json, indent=4)
    #with open(args.output, 'w') as f:
    #    print(j, file=f)

    write_json(ret_json, args.output)
