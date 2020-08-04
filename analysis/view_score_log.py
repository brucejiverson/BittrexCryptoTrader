import pandas as pd
from tools.tools import f_paths
df = pd.read_pickle(f_paths['score log'])
print(df)