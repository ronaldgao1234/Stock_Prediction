import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
def in_notebook(): return 'ipykernel' in sys.modules

# from tqdm import tqdm, tnrange
import tqdm as tq
from tqdm import tqdm_notebook, tnrange
if in_notebook():
    def tqdm(*args, **kwargs): return tq.tqdm_notebook(*
                                                       args, file=sys.stdout, **kwargs)

    def trange(*args, **kwargs): return tq.trange(*
                                                  args, file=sys.stdout, **kwargs)
else:
    from tqdm import tqdm, trange
    tnrange = trange
    tqdm_notebook = tqdm
