from imports import *


def plot_missing_in_features(*dataframes, **kwargs):
    '''
        Given pandas csv(s), plot the numbeer of missing values for each feature
    '''
    names = None
    if 'names' in kwargs:
        names = kwargs['names']
    for i, df in enumerate(dataframes):
        fig = plt.figure(i, figsize=(10, 2))
        missing = df.isnull().sum()
        missing.sort_values(inplace=True)
        ax = fig.add_subplot(missing.plot.bar())
        if names:
            ax.set_title(names[i])
    plt.show()


def plot_ohlc(*dataframes, Date='Date', Open='Open', High='High', Low='Low', Close='Close'):
    ''' Given pandas csv(s), plot OHLC chart
        https://www.techtrekking.com/how-to-plot-simple-and-candlestick-chart-using-python-pandas-matplotlib/
        It takes like 30 seconds to plot
        Args:
            dataframes (pandas DataFrame) : dataframe(s) to plot
            open  (str) : feature name in dataframe(s) that has opening price
            close (str) : same as open
            low   (str) : same as open
            high  (str) : same as open
    '''

    for i, df in enumerate(dataframes):
        # Converting date to pandas datetime format
        df_ = df.copy()
        df_['Date'] = pd.to_datetime(df_['Date'])
        df_["Date"] = df_["Date"].apply(mdates.date2num)

        # Creating required data in new DataFrame OHLC
        ohlc = df_[[Date, Open, High, Low, Close]].copy()
        f1, ax = plt.subplots(figsize=(10, 5))

        # plot the candlesticks
        candlestick_ohlc(ax, ohlc.values, width=.6,
                         colorup='green', colordown='red')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.show()


# class LivePlotNotebook(object):
#     """
#     Live plot using %matplotlib notebook in jupyter notebook
    
#     Usage:
#     ```
#     import time
#     liveplot = LivePlotNotebook()
#     x=np.random.random((10,))
#     for i in range(10):
#         time.sleep(1)
#         liveplot.update(
#             x=x+np.random.random(x.shape)/10,
#             actions=np.random.randint(0, 3, size=(10,))
#         )
#     ```
    
#     url:
#     """

#     def __init__(self):
#         %matplotlib notebook
#         fig,ax = plt.subplots(1,1)
        
#         ax.plot([0]*20, label='train')
#         ax.plot([0]*20, label='valid')
# #         ax.plot([0]*20, [1]*20, 'o', markersize=12,c='gray', label='hold')
# #         ax.plot([0]*20, [0]*20, '^', ms=12,c='blue', label='buy' )
# #         ax.plot([0]*20, [0]*20, 'v', ms=12,c='red', label='sell')
        
#         ax.set_xlim(0,1)
#         ax.set_ylim(0,1)
#         ax.legend()
#         ax.set_xlabel('epochs')
#         ax.grid()
#         ax.set_title('Losses')
        
#         self.ax = ax
#         self.fig = fig

#     def update(self, trn_loss, val_loss):             
#         # update price
#         trn_line = self.ax.lines[0]
#         trn_line.set_xdata(range(len(trn_loss)))
#         trn_line.set_ydata(trn_loss)
        
#         val_line = self.ax.lines[1]
#         val_line.set_xdata(range(len(val_loss)))
#         val_line.set_ydata(val_loss)
        
#         # update action plots
# #         for i, line in enumerate(self.ax.lines[1:]):
# #             line.set_xdata(np.argwhere(actions==i).T)
# #             line.set_ydata(x[actions==i])

#         # update limits
#         self.ax.set_xlim(0, len(trn_loss))
#         self.ax.set_ylim(min(min(trn_loss),min(val_loss)), max(max(trn_loss),max(val_loss)))

#         self.fig.canvas.draw()
        
# # Test
# # import time
# # liveplot = LivePlotNotebook()
# # x=[1]
# # x2=[2]
# # for i in range(10):
# #     time.sleep(1)
# #     print(x)
# #     x.append(random.randint(1,2))
# #     x2.append(random.randint(1,2))
# #     liveplot.update(
# #         trn_loss=x,
# #         val_loss=x2,
# #     )
