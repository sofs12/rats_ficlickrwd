import numpy as np
import matplotlib.pyplot as plt

def get_cp(cumsum_array, origin, transition = {'LH', 'HL'}):
      
    yy = cumsum_array[origin:] - cumsum_array[origin]
    xx = np.arange(0, len(yy), 1)   
    
    m = yy[-1]/xx[-1]
    yline = m*xx    
    
    if transition == 'LH': #low to high transition: want to look at the difference yline - yy; yline has positive slope
        cp = np.argmax(yline - yy)

    if transition == 'HL': # high to low transition: want to look at the difference yy - yline; yline has negative slope
        cp = np.argmax(yy - yline)

    return origin + cp

def LogOdds (N1, N2, p1, p2):
  r1 = p1/N1
  r2 = p2/N2
  r = (p1+p2)/(N1+N2)

  e = 0.00000001 # to prevent inf in the log
  
  result = p1 * np.log(r1+e) + p2 * np.log(r2+e) - (p1+p2) * np.log(r+e)

  return result

def accepted_cp_Gallistel(trial, theta, df, array_name, bool_onlyfirst = False, transition = 'LH', bool_plot = False):
    '''
    Return a tupple

    Parameters:
    trial
    theta
    df
    array_name
    bool_onlyfirst
    transition
    bool_plot

    '''
    # temporal resolution to 0.1s

    lvr = df.query(f'trialno == {trial}')[array_name].values[0]

    no_bins = int(np.ceil(lvr[-1]/100)) 

    last_bin = no_bins * 100
    lvr_binned = np.histogram(lvr, bins = no_bins, range = (0,last_bin))[0]
    yy = np.cumsum(lvr_binned)
    cp_all = []
    LogOdds_all = []
    cp_accepted= []
    origin = 0


    while origin < len(yy): 
        cp = get_cp(yy, origin, transition = transition)
        cp_all.append(cp)
        n1 = cp - origin 
        n2 = len(yy) - cp 
        if n1 == 0 or n2 == 0:
          break
        p1 = yy[cp] - yy[origin] 
        p2 = yy[-1] - yy[cp] 
        LogOdds_all.append(LogOdds(n1,n2,p1,p2))

        if LogOdds(n1, n2, p1, p2) > theta:
            cp_accepted.append(cp)
            origin = cp

        else:
            #don't accept change point and don't move origin here
            #and because the origin is not moved and the max deviation point will always be that one, this means there are no change points
            break

    cp_all = np.array(cp_all)/10 #back to s
    cp_accepted = np.array(cp_accepted)
    LogOdds_all = np.array(LogOdds_all)

    if len(cp_accepted) > 0:

        if bool_onlyfirst: # I think this is not correct - it is not always the case that the first cp is the one that divides the space the most (i.e. not the one with the highest LogOdds)
            cp_accepted = cp_accepted[np.argmax(LogOdds_all)]

        slices = np.hstack([0, cp_accepted, len(yy)-1])
        rate = np.zeros(len(slices)-1)
        for i in range(len(slices)-1):
          rate[i] = ( yy[slices[i+1]] - yy[slices[i]] ) / (slices[i+1] - slices[i])

        cp_accepted = cp_accepted/10
        rate = rate*10
        slices = slices/10

    else: 
        rate = np.nan
        slices = np.nan
        cp_accepted = np.nan


    if bool_plot:
        fig = plt.figure(figsize = (4,4), tight_layout = True)
        plt.title(f'trial {trial}, theta {theta}')
        xx = np.arange(0,len(yy))/10

        plt.plot(xx,yy)
        plt.plot([0,xx[-1]],[0,yy[-1]])

        for cp in cp_all:
          plt.plot(np.ones(2)*cp, [0,yy[-1]], alpha = 0.6, color = 'grey')
        for cp in [cp_accepted]:
          plt.plot(np.ones(2)*cp, [0,yy[-1]], alpha = 0.6, color = 'red')
        plt.ylabel('# cumulative')

        #plt.xlim(0)
        #plt.ylim(0)

        plt.savefig(f'trial{trial}_square_theta{theta}.png')
        plt.savefig(f'trial{trial}_square_theta{theta}.eps', format = 'eps')
        fig.show()

        #fig, axs = plt.subplots(2, 1, sharex='col', figsize = (10,8))
        #fig.suptitle(f'trial {trial}, theta {theta}')
        #xx = np.arange(0,len(yy))/10
        #axs[0].plot(xx,yy)
        #axs[0].plot([0,xx[-1]],[0,yy[-1]])
        #for cp in cp_all:
        #  axs[0].plot(np.ones(2)*cp, [0,yy[-1]], alpha = 0.6, color = 'grey')
        #for cp in [cp_accepted]:
        #  axs[0].plot(np.ones(2)*cp, [0,yy[-1]], alpha = 0.6, color = 'red')
        #axs[0].set_ylabel('# cumulative')
        #for i in range(len(rate)):
        #  axs[1].plot([slices[i],slices[i+1]], [rate[i], rate[i]], color = 'grey')
        #axs[1].set_xlabel('t (s)')
        #axs[1].set_ylabel('rate (Hz)')
        #fig.savefig(f'trial{trial}_theta{theta}.png')
        #fig.savefig(f'trial{trial}_theta{theta}.eps', format = 'eps')
        #fig.show()

    return cp_accepted, cp_all, LogOdds_all, rate, slices


def simple_accepted_cp_Gallistel(array, theta, bool_onlyfirst = False, transition = 'LH'):
    '''
    Return a tupple

    Parameters:
    trial
    theta
    df
    array_name
    bool_onlyfirst
    transition
    bool_plot

    '''
    # temporal resolution to 0.1s
    #array = int(array*1000) # to make sure we're dealing with integers - will divide back in the end

    no_bins = int(np.ceil(array[-1]/100)) 

    last_bin = no_bins * 100
    lvr_binned = np.histogram(array, bins = no_bins, range = (0,last_bin))[0]
    yy = np.cumsum(lvr_binned)
    cp_all = []
    LogOdds_all = []
    cp_accepted= []
    origin = 0

    while origin < len(yy): 
        cp = get_cp(yy, origin, transition = transition)
        cp_all.append(cp)
        n1 = cp - origin 
        n2 = len(yy) - cp 
        if n1 == 0 or n2 == 0:
          break
        p1 = yy[cp] - yy[origin] 
        p2 = yy[-1] - yy[cp] 
        LogOdds_all.append(LogOdds(n1,n2,p1,p2))

        if LogOdds(n1, n2, p1, p2) > theta:
            cp_accepted.append(cp)
            origin = cp

        else:
            #don't accept change point and don't move origin here
            #and because the origin is not moved and the max deviation point will always be that one, this means there are no change points
            break

    cp_all = np.array(cp_all)/10 #back to s
    cp_accepted = np.array(cp_accepted)
    LogOdds_all = np.array(LogOdds_all)

    if len(cp_accepted) > 0:

        if bool_onlyfirst: # I think this is not correct - it is not always the case that the first cp is the one that divides the space the most (i.e. not the one with the highest LogOdds)
            cp_accepted = cp_accepted[np.argmax(LogOdds_all)]

        slices = np.hstack([0, cp_accepted, len(yy)-1])
        rate = np.zeros(len(slices)-1)
        for i in range(len(slices)-1):
          rate[i] = ( yy[slices[i+1]] - yy[slices[i]] ) / (slices[i+1] - slices[i])

        cp_accepted = cp_accepted/10
        rate = rate*10
        slices = slices/10

    else: 
        rate = np.nan
        slices = np.nan
        cp_accepted = np.nan

    return cp_accepted, cp_all, LogOdds_all, rate, slices

def get_max_dev(cumsum_array, origin, last_point):
    yy = cumsum_array[last_point] - cumsum_array[origin]
    xx = np.arange(0, len(yy), 1)   
    
    m = yy[-1]/xx[-1]
    yline = m*xx    
    
    if transition == 'LH': #low to high transition: want to look at the difference yline - yy; yline has positive slope
        cp = np.argmax(yline - yy)

    if transition == 'HL': # high to low transition: want to look at the difference yy - yline; yline has negative slope
        cp = np.argmax(yy - yline)

    return origin + cp

#def proper_cp_Gallistel(array):
#   
#    no_bins = int(np.ceil(array[-1]/100)) 
#    last_bin = no_bins * 100
#    lvr_binned = np.histogram(array, bins = no_bins, range = (0,last_bin))[0]
#    yy = np.cumsum(lvr_binned)
#
#    origin = 0
#    for last_point in range(origin, len(array))
#        get_max_dev(array, 0, last_point)
#
#
##%%
###what if I do this using dataframes??
#
#bhvdf.query('bool_cp == True')
##%%
#
#array = bhvdf.loc[:,:,25].lever_rel.values[0]
## %%
#no_bins = int(np.ceil(array[-1]/100)) 
#last_bin = no_bins * 100
#lvr_binned = np.histogram(array, bins = no_bins, range = (0,last_bin))[0]
#yy = np.cumsum(lvr_binned)
##plt.plot(np.cumsum(array))
#
##%%
##array
#df = pd.DataFrame()
#df['timebin_start'] = np.arange(1,int(np.ceil(array[-1]/100)+1))
#df['events'] = 0
#df.loc[np.floor(array/100).astype(int), 'events'] = 1
#df['cumsum_events'] = df.events.cumsum()
##%%
##df['diff_to_origin_0'] = df.cumsum_events - df.cumsum_events[0]
##df['max_diff_origin_0']
##%%
#df['xx'] = df.apply(lambda x: np.arange(0,x.timebin_start), axis = 1)
#df['yy'] = df.xx.apply(lambda x: df.loc[x, 'cumsum_events'].values)
#
#df['yy_line'] = df.apply(lambda x: x.xx*x.yy[-1]/x.xx[-1], axis = 1)
#
#df['diff_line'] = df.yy_line - df.yy
#df['max_dev'] = df.apply(lambda x: x.xx[np.argmax(x.diff_line)], axis = 1)
##%%
#df['log_odds'] = df.apply(lambda x: LogOdds(x.xx[x.max_dev], x.xx[-1],x.yy[x.max_dev], x.yy[-1]), axis = 1)
##%%
#df['over_theta'] = df.log_odds > 2
##%%
#cps = []
#cps.append(df.query('over_theta').max_dev.unique()) # -- there is only one point. what happens when we have multiple? does it happen?+-
##%%
##ok, so now we have a log_odds that is always the same when starting at the origin, great!
##the next step is moving the origin to this point and sweeping through the rest of the space
#
### maybe do a different pandas for each origin?
#cps
##%%
#df.loc[cps[0]-1]
## %%
#plt.plot(df.cumsum_events)
## %%
#df.loc[[1,2], 'cumsum_events']
## %%
#plt.plot(df.loc[159,'yy_line'])
## %%
#

def validate_cp(cp, lever_array):

    if len(lever_array)<3:
        cp = np.nan

    if round(lever_array[-1]/1000,0) == round(cp,0):
        cp = np.nan

    return cp