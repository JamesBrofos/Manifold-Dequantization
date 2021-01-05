import glob
import os
import re


def get_numbers(s):
    f = re.findall('\d*\.?\d+', s)
    if len(f) > 0:
        return float(f[0])
    else:
        return None

def value_or_nan(s):
    if s is not None:
        return s
    else:
        return 'NaN'

class Results:
    def __init__(self, method, mean_mse, cov_mse, klqp, klpq, ress):
        self.method = method
        self.mean_mse = mean_mse
        self.cov_mse = cov_mse
        self.klqp = klqp
        self.klpq = klpq
        self.ress = ress

    def __str__(self):
        return '{},{},{},{},{},{}'.format(self.method,
                                          value_or_nan(self.mean_mse),
                                          value_or_nan(self.cov_mse),
                                          value_or_nan(self.klqp),
                                          value_or_nan(self.klpq),
                                          value_or_nan(self.ress))

    @classmethod
    def from_file(cls, file):
        for line in file:
            if 'Mean' in line:
                ls = line.strip().split('-')
                method = ls[0].strip()
                mean_mse = get_numbers(ls[1])
                cov_mse = get_numbers(ls[2])
                klqp = get_numbers(ls[3])
                if len(ls) == 6:
                    klpq = get_numbers(ls[4])
                else:
                    klpq = None
                ress = get_numbers(ls[-1])
                return cls(method, mean_mse, cov_mse, klqp, klpq, ress)


print('method,mean_mse,cov_mse,klqp,klpq,ress')
odir = os.path.join('..', 'output', '*.log')
files = glob.glob(odir)

for fn in files:
    with open(fn, 'r') as f:
        print(Results.from_file(f))
