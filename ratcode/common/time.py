## binning, aligment helpers, etc

def convert_date_bonsai(date):
    if len(date) == 6:
        bonsai_date = date[:2] + '-' + date[2:4] + '-' + date[4:]
    else:
        print('check format; this function accepts YYMMDD')
        bonsai_date = 0
    return bonsai_date

