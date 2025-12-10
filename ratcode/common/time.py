## binning, aligment helpers, etc
import scipy

def convert_date_bonsai(date):
    if len(date) == 6:
        bonsai_date = date[:2] + '-' + date[2:4] + '-' + date[4:]
    else:
        print('check format; this function accepts YYMMDD')
        bonsai_date = 0
    return bonsai_date

def convert_timestamp(time_A, ref_A, ref_B):

    '''
    Converts time_A into time_B (A the old, B the new time frame)
    time_A: time to convert; can be a float or array
    ref_A: set of values in the original time reference
    ref_B: set of values in the new reference frame
    '''

    regress = scipy.stats.linregress(ref_A, ref_B)

    time_B = regress.intercept + regress.slope * time_A

    return time_B