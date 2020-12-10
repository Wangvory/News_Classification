import os
import calendar
from datetime import datetime
import calendar

        
def split_sources(arti):

        data1 = arti.split('\t')

        if data1[0][0] == "\"" and data1[0][-1] == "\"":
            date = data1[0][1:-1]
        else: 
            date = data1[0]
        date = date[:10]
        month, day, year = date.split('/')
        month = calendar.month_name[int(month)]
        source = data1[2].split(" ")[0]
                               
        if not os.path.exists('..\\boto3\{}\{}\{}'.format(year,month,day)):
            os.makedirs('..\\boto3\{}\{}\{}'.format(year,month,day))
        if source == "" or source == " " or '(No' in source:
                source = 'Unknown or Corrupt source name'
            
        with open('..\\boto3\{}\{}\{}\{}.tsv'.format(year,month,day,source), 'a', encoding='utf8') as wf:
            print("split_db "+' ', source)
            wf.write(arti+'\n')