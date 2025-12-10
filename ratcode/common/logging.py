''' data.py
This module has all the functions needed to parse the behavioural data
'''
import os
import time

def getEventCodes(file_to_parse):
    """
    Return two dictionaries: codename_dict and codename_dict_reverse
    codename_dict: keys are events names; values are corresponding event numbers
    codename_dict_reverse: keys and values are swaped

    Parameters:
    file_to_parse: file with the correspondence between event names and numbers
    """

    codename_dict = dict()
    codename_dict_reverse = dict()

    f = open(file_to_parse, "r")

    for line in f.readlines():
        upper_index = 100
        isUpperInLine = True

        x = line.split()
        i=0

        for word in x:
            if(word.isupper()):
                upper_index = i
            i+=1
            if(word == x[-1] and upper_index == 100):
                isUpperInLine = False

        if(isUpperInLine==True):

            if(upper_index == 3 and x[upper_index +1] == '='):
                if(x[upper_index+2][-1] == ";"):
                    x[upper_index+2] = x[upper_index +2][:-1]

                codename_dict[x[upper_index +2]] = x[upper_index]
                codename_dict_reverse[x[upper_index]] = x[upper_index +2]

            else:
                print("line not following expected structure: " , line)

    return codename_dict, codename_dict_reverse


def getSessionVariables(bhv_log_file):
    #animal ID, date, starting time, experimenter, FI, RWD (fixed or varying) and Click (y/n)
    
    x = bhv_log_file.split('\\') #goes to the last directory
    x = x[-1].split('.') #gets rid of the file extension
    x = x[0].split('_')

    animalID = x[0]
    date = x[2]
    experimenter = x[3]

    timeFileCreated = time.ctime(os.path.getctime(bhv_log_file))
    timeFileCreated = timeFileCreated.split()[3]

    # to include training stages, where these variables are not defined
    FISession = 0
    clickSession = 0
    rwdrateSession = 0
    rwdMod = 1

    f = open(bhv_log_file, 'r') 
    for line in f.readlines():
        eventcode = line.split()[0] # looking for session variables, with codes 30's
        if eventcode == '30': # FI session
            FISession = line.split()[1]
        if eventcode == '31': # click
            clickSession = line.split()[1]
        if eventcode == '32': # reward rate
            rwdrateSession = line.split()[1]
        if eventcode == '33': # reward modifier
            rwdMod = line.split()[1]


    return animalID, date, timeFileCreated, experimenter, FISession, clickSession, rwdrateSession, rwdMod


def splitLogFile(bhv_log_file, destination_folder):

    print('splitting..')

    session_vars = []
    actual_events = []

    f = open(bhv_log_file, 'r')
    for line in f.readlines():
        if len(line.split()) > 1:
        #if line.split()[0][0] == '3':
            if line.split()[0] in ('30','31','32', '33', '148', '114'):
                session_vars.append(line)
            else:
                actual_events.append(line)
        else:
            print(line)
    
    file_prefix = bhv_log_file.split('.')[0].split('\\')[-1]
    file_prefix = os.path.join(destination_folder, file_prefix)               
    file_vars = file_prefix + '_temp_vars.txt'
    file_events = file_prefix + '_temp_events.txt'

    file_vars = os.path.join(destination_folder, file_vars)
    file_events = os.path.join(destination_folder, file_events)

    f_vars = open(file_vars, 'w')
    for entry in session_vars:
        f_vars.write(entry)
    f_vars.close()

    f_events = open(file_events, 'w')
    for entry in actual_events:
        f_events.write(entry)
    f_events.close()

    return file_prefix, file_vars, file_events

def determine_experiment(df):
    FIs = len(df.FI.unique())
    rwds = len(df.n_protocols.unique())

    if FIs == 3 and rwds == 1:
        exp = 'a'

    elif FIs == 3 and rwds == 3:
        exp = 'b'

    elif FIs == 1 and rwds == 3:
        exp = 'c'
    
    else:
        exp = 'undetermined'

    return exp

def determine_experiment(df):
    FIs = len(df.FI.unique())
    rwds = len(df.n_protocols.unique())

    if FIs == 3 and rwds == 1:
        exp = 'a'

    elif FIs == 3 and rwds == 3:
        exp = 'b'

    elif FIs == 1 and rwds == 3:
        exp = 'c'
    
    else:
        exp = 'undetermined'

    return exp