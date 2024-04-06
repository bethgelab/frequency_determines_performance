import os

for f in os.listdir('.'):
    if '_results.json' in f:
        os.rename(f, f.replace('results', 'ensemble_results'))