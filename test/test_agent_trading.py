import pandas as pd
from agents.agents import TestLogic
import matplotlib.pyplot as plt

# Make the data
d = {'BTCClose': [x/10 for x in [*range(1000, 1100, 1), *range(1100, 900, -1)]]}
f_map = {'BTCClose': 0}
action_size = 1

agent = TestLogic(f_map, action_size)

for i in [0,1,2,3]:
    if i == 0:
        agent.last_action = 1
        agent.trailing_stop_loss['percent'] = 1
        title = 'Trailing stop loss'
    if i == 1:
        agent.trailing_stop_order['percent'] = 1
        title = 'Trailing stop order'
    if i == 2:
        agent.stop_loss = 95
        title = 'Stop loss'
    if i == 3:
        agent.stop_order = 105
        title = 'Stop order'
    actions = []
    for item in d['BTCClose']:
        state = [item]
        action = agent.act(state)
        actions.append(action)

    d['Actions'] = actions
    df = pd.DataFrame(d)

    fig, ax = plt.subplots(1,1)
    plt.title(title)
    df.plot(y='BTCClose', ax=ax)
    df.plot(y='Actions', ax=ax)
plt.show()