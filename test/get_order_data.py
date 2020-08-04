from environments.environments import BittrexExchange
import pandas as pd

env = BittrexExchange(verbose=3)
env.get_and_save_order_history()

