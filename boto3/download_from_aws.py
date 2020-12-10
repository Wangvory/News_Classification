import os, sys
import glob
import boto3
import calendar
from datetime import datetime,timedelta


sys.path.append(os.path.abspath("..\cache"))
from window import *

s3 = boto3.client('s3')

def download_prev_seen(dates):
    last =''
    for d in dates:
        try:
            if s3.head_object(Bucket='privco-news-collection', Key='Database_with_label/Previously_seen/previously_seen_{}.txt'.format(d)):
                last = d
                break 
        except:
            continue    
    file_in_s3 = 'previously_seen_{}.txt'.format(last)
    if not os.path.exists('..\cache\previously_seen'):
        os.makedirs('..\cache\previously_seen')
    file_save_as = '..\cache\previously_seen\\'+file_in_s3
    s3.download_file('privco-news-collection', 'Database_with_label/Previously_seen/'+file_in_s3, file_save_as)
    return last


def download_database(today):

    year,month,day  = today.split('-')
    month = calendar.month_name[int(month)]
    # print(day,month,year)
    length = len('Database_with_label/Database/'+year+'/'+month+'/'+day+'/')
    for key in s3.list_objects(Bucket='privco-news-collection')['Contents']:

        if 'Database_with_label/Database/'+year+'/'+month+'/'+day+'/' in (key['Key']):
            # print(key)
            file_save_as = key['Key'][length:]
            
            if not os.path.exists('..\\boto3\\{}\\{}\\{}'.format(year,month,day)):
                os.makedirs('..\\boto3\\{}\\{}\\{}'.format(year,month,day)) 
            save = '..\\boto3\\{}\\{}\\{}\{}'.format(year,month,day,file_save_as)
            s3.download_file('privco-news-collection', key['Key'], save) 

    try:
        file_in_s3 = 'database_{}.tsv'.format(today)
        bankruptcy_ipo_file_in_s3 = 'bankruptcy_ipo_{}.tsv'.format(today)
        
        if s3.head_object(Bucket='privco-news-collection', Key='Database_with_label/Daily_consolidated_database/Bankruptcy&IPO/'+bankruptcy_ipo_file_in_s3):
            if not os.path.exists('..\database'):
                os.makedirs('..\database')
            file_save_as = '..\database\\bankruptcy_ipo_{}.tsv'.format(today)
            s3.download_file('privco-news-collection', 'Database_with_label/Daily_consolidated_database/Bankruptcy&IPO/'+ bankruptcy_ipo_file_in_s3, file_save_as)

        if s3.head_object(Bucket='privco-news-collection', Key='Database_with_label/Daily_consolidated_database/'+file_in_s3):
            file_save_as = '..\database\database_{}.tsv'.format(today)
            s3.download_file('privco-news-collection', 'Database_with_label/Daily_consolidated_database/'+file_in_s3, file_save_as)
            
            first_time_today = False
    except Exception as e:
        print(e)
        first_time_today =True 
    return first_time_today


def download_from_s3():
    
    today = str(datetime.today())[:10]
    dates = [str(datetime.today() - timedelta(days=x))[:10] for x in range(0,4)]
    last = download_prev_seen(dates)
    first_time_today = download_database(today)
    create_todays_prev_seen(last)
    return first_time_today

# download_from_s3()