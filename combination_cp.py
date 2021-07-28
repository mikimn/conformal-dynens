import script.utils as cf


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--indexfile", help="index file", required=False)
parser.add_argument("-d", "--workdir", help="work directory", required=True)
parser.add_argument("-a", "--alpha", help="alpha parameter for ensemble prediction set", required=True, default=0.2, type=float)

args = parser.parse_args()

workdir = args.workdir
indexfile = args.indexfile
# Ensemble prediction set consensus strength
alpha = args.alpha

if not indexfile:
    indexfile = f'{workdir}/index_cp.csv'

print('Combining...')
combination = cf.Combination()
combination.get_config(indexfile)
combination.read_model_outputs('')
final1, sums = combination.conformal_selection(alpha=alpha)

final1.to_csv(f'{workdir}/prediction_set_conformal_threshold_selection_alpha_{alpha}.csv', header=True, index=False)
# sums.to_csv(f'{workdir}/sums_conformal_threshold_selection.csv', header=True, index=False)

