## subjects lists, batches, etc


"""
.##....##..#######.....########..##........#######...######..##....##..######.
.###...##.##.....##....##.....##.##.......##.....##.##....##.##...##..##....##
.####..##.##.....##....##.....##.##.......##.....##.##.......##..##...##......
.##.##.##.##.....##....########..##.......##.....##.##.......#####.....######.
.##..####.##.....##....##.....##.##.......##.....##.##.......##..##.........##
.##...###.##.....##....##.....##.##.......##.....##.##....##.##...##..##....##
.##....##..#######.....########..########..#######...######..##....##..######.
"""

animal_order = ['Hydrogen', 'Helium', 'Lithium', 'Berylium']
animal_order2 = ['Boron', 'Carbon', 'Nitrogen', 'Oxygen']
animal_order3 = ['Fluorine', 'Neon', 'Sodium', 'Magnesium']

FI_order = [30, 60]
click_order = [False, True]
#rwd_order = [15,30] #only for the pumps no blocks




animal_order_full = animal_order + animal_order2 + animal_order3

session_vars_order = ['FI 30 click False',
                      'FI 30 click True',
                      'FI 60 click False',
                      'FI 60 click True']

conditions_order = ['FI 30\nclick False\nrwd 1.5',
                    'FI 30\nclick False\nrwd 3.0',
                    'FI 30\nclick True\nrwd 1.5',
                    'FI 30\nclick True\nrwd 3.0',
                    'FI 60\nclick False\nrwd 1.5',
                    'FI 60\nclick False\nrwd 3.0',
                    'FI 60\nclick True\nrwd 3.0']



#dictionary for rwd rate
#reward modifier to rwd amount (uL)
rwd_dic = dict()
rwd_dic[1.5] = 37.5
rwd_dic[3] = 75

#eliminate data from when timeout was still in place; the ones after July 6th are
#from experimenting with blocks // changing rewards mid session
dates_to_drop = ['210202', '210203', '210204', '210206', '210207', '210208',
       '210209', '210210', '210211', '210212', '210215', '210216',
       '210217', '210218', '210219', '210222', '210223', '210224', '210225',
       '210226', '210301', '210302', '210303', '210304', '210305',
       '210308', '210309', '210310', '210311', '210312', '210315',
       '210316', '210317', '210318', '210319', '210322', '210323',
       '210324', '210325', '210326', '210923', '210924', '210925', '210927',
       '210928', '210929', '210930', '211001', '211004', '210915', '210916', '210917',
       '210920', '210921', '210923', '210924', '210925', '210927',
       '210928', '210929', '210930', '211001', '211004', '211005',
       '211006', '211007', '211019', '211020', '211021', '211022',
       '211025', '211026', '211027', '211028', '211102', '211103',
       '211104', '211105', '211108', '211109', '211110',
       '220131', '220223']


#dates to eliminate if animal == Hydrogen (click sessions with only one speaker)
#16 june is the first date with the new box
H_dates_to_drop = ['210405', '210406', '210407', '210408', '210409', '210412',
       '210413', '210414', '210415', '210416', '210419', '210511',
       '210512', '210513', '210514', '210517', '210518', '210519',
       '210603', '210604', '210607']

Na_dates_to_drop = ['220311','220314','220331']


"""
.########..##........#######...######..##....##..######.
.##.....##.##.......##.....##.##....##.##...##..##....##
.##.....##.##.......##.....##.##.......##..##...##......
.########..##.......##.....##.##.......#####.....######.
.##.....##.##.......##.....##.##.......##..##.........##
.##.....##.##.......##.....##.##....##.##...##..##....##
.########..########..#######...######..##....##..######.
"""

animal_order3 = ['Fluorine', 'Neon', 'Sodium', 'Magnesium']
animal_order4 = ['Phosphorus', 'Sulfur', 'Chlorine', 'Argon']
animal_order5 = ['Potassium', 'Calcium', 'Scandium', 'Titanium']


nprots_order_blocks = [7,14,28]



experiment_FI_dic = {
    'a': [15,30,60],
    'b': [15,30,60],    
    'c': [30],
    'other': []
}

experiment_nprots_dic = {
    'a': [14],
    'b': [7,14,28],    
    'c': [7,14,28],
    'other': []
}


FI_order_blocks = [15,30,60]
exp_list = ['a', 'b', 'c']

# experiments title
experiment_title_dic = {
    'a': 'change FI, reward 14',
    'b': 'matched reward rate',
    'c': 'change reward, FI 30'
}




animal_order_blocks_click = ['Sodium', 'Magnesium', 'Chlorine', 'Argon', 'Scandium', 'Titanium']

extra_conds_dic = {
    'a': 'experiment == "a" and FI_len == 3 and nprots == 14',
    'b': 'experiment == "b" and FI_len == 3 and nprots_len == 3',
    'c': 'experiment == "c" and FI == 30 and nprots_len == 3'
}

dodge_dic = {
    'a': True,
    'b': False,
    'c': False
}