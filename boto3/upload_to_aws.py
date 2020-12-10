import os
import time
import glob
import boto3
import calendar
from datetime import datetime
s3 = boto3.resource('s3')    


def upload_previously_seen(to_delete,today):
    if os.path.isfile('..\cache\previously_seen\previously_seen_{}.txt'.format(today)):
        prev_file = '..\cache\previously_seen\previously_seen_{}.txt'.format(today)
        previously_seen = open(prev_file, 'rb')
        s3.Bucket('privco-news-collection').put_object(Key='Database_with_label/Previously_seen/'+prev_file[-30:], Body=previously_seen)
        for filename in os.listdir('..\cache\previously_seen\\'):
            to_delete.append('..\cache\previously_seen\\'+filename)
        

def upload_database(to_delete,today):
    year,month,day = today.split('-')
    month = calendar.month_name[int(month)]
    
    if os.path.isfile('..\database\database_{}.tsv'.format(today)):
        complete_db_file = '..\database\database_{}.tsv'.format(today)
        database_complete = open(complete_db_file, 'rb')
        s3.Bucket('privco-news-collection').put_object(Key='Database_with_label/Daily_consolidated_database/'+complete_db_file[-23:], Body=database_complete)
        to_delete.append(complete_db_file)

    if os.path.isfile('..\database\\bankruptcy_ipo_{}.tsv'.format(today)):
        bankruptcy_ipo_db_file = '..\database\\bankruptcy_ipo_{}.tsv'.format(today)
        database_complete = open(bankruptcy_ipo_db_file, 'rb')
        s3.Bucket('privco-news-collection').put_object(Key='Database_with_label/Daily_consolidated_database/Bankruptcy&IPO/'+bankruptcy_ipo_db_file[-29:], Body=database_complete)
        to_delete.append(bankruptcy_ipo_db_file)

    for filename in os.listdir('..\\boto3\{}\{}\{}'.format(year,month,day)):
        db_file = '..\\boto3\{}\{}\{}\\{}'.format(year,month,day,filename)
        database = open(db_file, 'rb')
        s3.Bucket('privco-news-collection').put_object(Key='Database_with_label/Database/{}/{}/{}/{}'.format(year,month,day,filename), Body=database)
        to_delete.append(db_file)
    

def upload_to_s3(today):
    to_delete = []
    year,month,day = today.split('-')
    month = calendar.month_name[int(month)]
    upload_previously_seen(to_delete,today)
    upload_database(to_delete,today)
    for x in to_delete:
        if os.path.isfile(x):
            os.remove(x)

    os.rmdir('..\cache\previously_seen')
    os.rmdir('..\database')
    os.rmdir('..\\boto3\{}\{}\{}'.format(year,month,day))
    os.rmdir('..\\boto3\{}\{}'.format(year,month))
    os.rmdir('..\\boto3\{}'.format(year))
    

    
    
    