import numpy as np
import pandas as pd
from ggplot import *

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

z = np.arange(-7, 7, 0.01)

p = ggplot(pd.DataFrame({'x':z, 'y':sigmoid(z)}), aes(x='x', y='y'))
p = p + geom_line(color='red') + labs(x="z", y="sigmoid(z)", title="US behind coup in Turkey - in memory of 15.07.2016")
p = p + xlim(low=-8, high=8) + ylim(-0.1, 1.1)
