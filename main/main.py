import sys 
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
os.chdir('C:\Users\jzhou\iCloudDrive\Desktop\Newsfeeds_Labelling-master')
sys.path.append(os.path.abspath('boto3'))
# from upload_to_aws import *
# from download_from_aws import *

sys.path.append(os.path.abspath('source_scripts'))
from axios import *
from finSME import *
from prnews import *
from VCNews import *
from cb_news import *
from fortune import *
from benzinga import *
from vcaonline import *
from pehub_new import *
from techcrunch import *
from venturebeat import *
from businesswire import *
from globenewswire import *
from PEprofessional import *
from businessjournals import *

today_date = str(datetime.now())[:10]
first_time_today = False
data_set = set()
if not os.path.exists('..\database'):
    os.makedirs('..\database')
if not os.path.exists('..\cache\previously_seen'):
    os.makedirs('..\cache\previously_seen')
filename = '..\cache\previously_seen\previously_seen_{}.txt'.format(today_date)
database = '..\database\database_{}.tsv'.format(today_date)
bankruptcy_ipo = '..\database\\bankruptcy_ipo_{}.tsv'.format(today_date)
# handles the 5 days window for duplicate deletion (in already_seen file)
if not os.path.isfile(filename):
    first_time_today = download_from_s3()
    
if os.path.isfile(filename):
    file = open(filename, 'r') 
    data = file.read()
    data = data.split('\n')[:-1]
    # for tab seperated inputs 
    for x in data:
        x = x[42:]
        data_set.add(x)
    file.close()
    
if not os.path.isfile(database):
        file = open(database, 'w')
        file.write("Date_Collected\tDate_Published\tSource\tArticle_Name\tArticle_Link\tDescription\t\
                    Article_Name_label\tArticle_Description_label\tER_from_Article_Name(Spacy)\tER_from_Article_Name(AWS-Comprehend)\n")
        file.close() 
if not os.path.isfile(bankruptcy_ipo):
        file = open(bankruptcy_ipo, 'w')
        file.write("Date_Collected\tDate_Published\tSource\tArticle_Name\tArticle_Link\tDescription\t\
                    Article_Name_label\tArticle_Description_label\tER_from_Article_Name(Spacy)\tER_from_Article_Name(AWS-Comprehend)\n")
        file.close() 

# calling all source scripts sequentially
main_FinSME(data_set,today_date,filename,database)
main_businesswire(data_set,today_date,filename,database)
main_techcrunch(data_set,today_date,filename,database)
main_VCN(data_set,today_date,filename,database)
main_venturebeat(data_set,today_date,filename,database)
main_prnews(data_set,today_date,filename,database)
main_businessjournals(data_set,today_date,filename,database)
main_vcaonline(data_set,today_date,filename,database)    
main_benzinga(data_set,today_date,filename,database)
main_globenewswire(data_set,today_date,filename,database)
main_pehub_new(driver,data_set,today_date,filename,database)

if 11/10/2020 == True:
    main_fortune(driver,today_date,database)
    main_pep(driver,data_set,today_date,filename,database)
    main_cb_news(driver,data_set,today_date,filename,database)
    main_axios(driver,data_set,today_date,filename,database)    

upload_to_s3(today_date)
driver.quit()