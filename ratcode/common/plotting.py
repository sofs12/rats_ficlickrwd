import matplotlib.pyplot as plt
import matplotlib

plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes
plt.rc('axes.spines', top = False, right = False) #equivalent to sns.despine(top = True)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.titlesize'] = 'large'


click_order = [False, True]
click_palette = ['#bababa', '#7a7a7a'] #following the order of the click_order

rwd_dict = { 7: '#CCE3DE',
            14: '#A4C3B2',
            28: '#6B9080'}

FI_dict = {15: '#cba6e3',
              30: '#81a6fc',
              60: '#77d674'}


#color_pal_dic = {
#    "FI 30 click False": "#81a6fc",
#    "FI 30 click True": "#2e6dff",
#    "FI 60 click False": "#77d674",
#    "FI 60 click True": "#0b6b07"
#}

#conds_block_order = list(color_blocks_dic.keys())

#color_pal = []
#for cond in session_vars_order:
#    color_pal.append(color_pal_dic[cond])

#color palette for different rewards (1st is the smaller rwd)
newc = ['#FF7E6B','#8C5E58']


color_nprots_blocks = ["#636f82", "#81a6fc", "#2e6dff"]

color_blocks_dic = {
    "FI15 rwd7": "#cba6e3",
    "FI15 rwd14": "#b979e0",
    "FI15 rwd15": "#b979e0",
    "FI15 rwd28": "#980cf0",
    "FI15 rwd30": "#980cf0",
    "FI30 rwd7": "#879aad",
    "FI30 rwd14": "#81a6fc",
    "FI30 rwd15": "#81a6fc",
    "FI30 rwd28": "#2e6dff",
    "FI30 rwd30": "#2e6dff",
    "FI30 rwd60": "#2e6dff", # shouldn't exist
    "FI60 rwd7": "#87ad93",    
    "FI60 rwd14": "#77d674",
    "FI60 rwd15": "#77d674",
    "FI60 rwd28": "#0b6b07",
    "FI60 rwd30": "#0b6b07"
}

agg_blocks_FI_dic = {
    15: "#cba6e3",
    30: "#81a6fc",
    60: "#77d674"
}

color_FI_blocks = list(agg_blocks_FI_dic.values())
