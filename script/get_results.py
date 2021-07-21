from collections import defaultdict
import pandas as pd
import numpy as np
import operator
import pickle
import utils as cf

import argparse


def compute_coverage_and_set_size(predictions_file, targets_file):
    # load ground truth file
    df_g = pd.read_csv(targets_file, header=0)
    # del df_g['9']  # Quickfix
    df_p = pd.read_csv(predictions_file, header=0)
    # del df_p['9']  # Quickfix
    coverage = (df_p.astype('bool') & df_g.astype(
        'bool')).sum(axis='columns').mean()
    avg_set_size = df_p.sum(axis='columns').mean()
    return coverage, avg_set_size


def compute_consistency(predictions_file_1, predictions_file_2):
    df_p1 = pd.read_csv(predictions_file_1, header=0).astype('bool')
    df_p2 = pd.read_csv(predictions_file_2, header=0).astype('bool')

    sz = min(len(df_p1), len(df_p2))
    df_p1 = df_p1.iloc[:sz]
    df_p2 = df_p2.iloc[:sz]

    intersection = (df_p1 & df_p2).sum(axis='columns')
    union = (df_p1 | df_p2).sum(axis='columns')
    return (intersection / union).mean()


def compute_correct_consistency(predictions_file_1, predictions_file_2, targets_file):
    df_g = pd.read_csv(targets_file, header=0).astype('bool')
    df_p1 = pd.read_csv(predictions_file_1, header=0).astype('bool')
    df_p2 = pd.read_csv(predictions_file_2, header=0).astype('bool')

    sz = min(len(df_p1), len(df_p2))
    df_p1 = df_p1.iloc[:sz]
    df_p2 = df_p2.iloc[:sz]

    intersection = (df_p1 & df_p2).sum(axis='columns')
    union = (df_p1 | df_p2).sum(axis='columns')
    return (intersection / union).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p1", "--predfile1",
                        help="prediction file1", required=True)
    parser.add_argument("-p2", "--predfile2",
                        help="prediction file2", required=True)
    parser.add_argument("-g", "--groundtruth",
                        help="ground truth file", required=True)
    parser.add_argument("-o", "--outputfile", help="output file")

    args = parser.parse_args()

    predfile1 = args.predfile1
    predfile2 = args.predfile2
    groundfile = args.groundtruth
    outputfile = args.outputfile

    if outputfile:
        f = open(outputfile, 'a')
        f.write('{},{},{},{},{},{},{},{},{}\n'.format(
            predfile1, predfile2, er, crl, er_ea, pearson, cosine, ea_er, cr))
        f.close()
