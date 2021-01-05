import numpy as np
import pandas as pd


data = pd.read_csv('results.csv')

mean = data.groupby('method').mean()
std = data.groupby('method').std() / np.sqrt(data.groupby('method').count())

for idx in mean.index:
    gm = mean.loc[idx]
    gs = std.loc[idx]
    s = ''
    for m in gm.index:
        s += ' & {:.4f} $\pm$ {:.4f}'.format(gm.loc[m], gs.loc[m])
    s = idx.title() + s + ' \\\\'
    s = s.strip()
    print(s)
