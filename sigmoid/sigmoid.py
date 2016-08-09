import numpy as np
import pandas as pd
from ggplot import *

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

z = np.arange(-7, 7, 0.01)

p = ggplot(pd.DataFrame({'x':z, 'y':sigmoid(z)}), aes(x='x', y='y'))
p = p + geom_line() + xlab('z') + ylab('sigmoid(z)')
p = p + xlim(low=-8, high=8) + ylim(-0.1, 1.1)
