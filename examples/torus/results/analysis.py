import numpy as np
import pandas as pd


data = pd.read_csv('results.csv')


for density in data['density'].unique():
    df = data.loc[data['density'] == density]
    gb = df.groupby(['method'])
    mean = gb.mean()
    std = gb.std() / np.sqrt(gb.count())

    for idx in mean.index:
        gm = mean.loc[idx]
        gs = std.loc[idx]
        s = ''
        for m in gm.index:
            s += ' & {:.4f} $\pm$ {:.4f}'.format(gm.loc[m], gs.loc[m])
        s = ' & ' + idx.title() + s + ' \\\\'
        if idx == 'direct':
            s = density.title() + s
        s = s.strip()
        print(s)
    print()
