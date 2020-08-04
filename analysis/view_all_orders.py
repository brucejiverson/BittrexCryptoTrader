import pandas as pd
import numpy as np
from tools.tools import f_paths

path = f_paths['order log']
df = pd.read_pickle(path)

print('ORDER DATA: ')
print(df)

# Drop some columns for easy viewing
# df.drop(columns=['OrderUuid'], inplace=True)
# df.to_pickle(path)
print(df.columns)
