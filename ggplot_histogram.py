import numpy as np
import pandas as pd
from ggplot import *

mu = 0
sigma = 3
size = 10000
x = np.random.normal(mu,sigma,size)

df = pd.DataFrame(x,columns=['x'])

p = ggplot(aes(x='x'), data=df)
p + geom_histogram(bins=100)