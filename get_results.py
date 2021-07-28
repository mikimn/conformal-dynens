import pandas as pd
import argparse
import csv
import os 


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
    # df_p1 = df_p1.iloc[-sz:]
    # df_p2 = df_p2.iloc[-sz:]

    intersection = (df_p1 & df_p2).sum(axis='columns')
    union = (df_p1 | df_p2).sum(axis='columns')
    # print(intersection)
    # print(union)
    # print(intersection / union)
    return (intersection / union).mean()


def compute_correct_consistency(predictions_file_1, predictions_file_2, targets_file):
    df_g = pd.read_csv(targets_file, header=0).astype('bool')
    df_p1 = pd.read_csv(predictions_file_1, header=0).astype('bool')
    df_p2 = pd.read_csv(predictions_file_2, header=0).astype('bool')

    sz = min(len(df_p1), len(df_p2))
    df_p1 = df_p1.iloc[:sz]
    df_p2 = df_p2.iloc[:sz]
    # df_p1 = df_p1.iloc[-sz:]
    # df_p2 = df_p2.iloc[-sz:]

    intersection = (df_p1 & df_p2 & df_g).sum(axis='columns')
    # print(intersection)
    ccon = intersection.mean()
    return ccon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p1", "--predfile1",
                        help="prediction file1", required=True)
    parser.add_argument("-p2", "--predfile2",
                        help="prediction file2", required=False)
    parser.add_argument("-g", "--groundtruth",
                        help="ground truth file", required=True)
    parser.add_argument("-o", "--outputfile", help="output file", required=False)

    args = parser.parse_args()

    predfile1 = args.predfile1
    predfile2 = args.predfile2
    groundfile = args.groundtruth
    outputfile = args.outputfile

    if args.predfile2:
        con = compute_consistency(predfile1, predfile2)
        print(f'Consistency: {con}')

        ccon = compute_correct_consistency(predfile1, predfile2, groundfile)
        print(f'Correct Consistency: {ccon}')

    coverage, set_size = compute_coverage_and_set_size(predfile1, groundfile)
    print(f'Coverage #1: {coverage}')
    print(f'Set Size #1: {set_size}')

    if args.predfile2:
        coverage2, set_size2 = compute_coverage_and_set_size(predfile2, groundfile)
        print(f'Coverage #2: {coverage2}')
        print(f'Set Size #2: {set_size2}')

    if args.outputfile:
        write_header = False
        if not os.path.exists(args.outputfile):
            write_header = True
        with open(args.outputfile, 'a') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['p1', 'p2', 'groundfile', 'con', 'ccon', 'coverage1', 'set_size1', 'coverage2', 'set_size2'])
            writer.writerow([predfile1, predfile2, groundfile, con, ccon, coverage, set_size, coverage2, set_size2])