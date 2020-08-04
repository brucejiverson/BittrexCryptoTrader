import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from agents.agents import logic_tester

price = [*range(9000, 10000, 10), *range(10000, 9000, 10)]


print(s.head())

f_map = {'BTCClose': 0}
size = 1
a = logic_tester()

actions = []
for p in price:
    action = logic_tester.act([p])
    actions.append(action)

df = pd.DataFrame(columns=['BTCClose', 'Actions'], 
                    data = [price, actions])
fig, axes = plt.subplots(2,1, sharex=True)
df.plot(y = 'BTCClose',ax=axes[0])
df.plot(y = 'Actions',ax=axes[1])
