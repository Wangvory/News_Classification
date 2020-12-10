import sys
import os
from datetime import datetime
from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# Windows
os.chdir(r'C:\Users\jzhou\iCloudDrive\Desktop\PrivCo\Newsfeeds_Labelling-master')
# Mac
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
filename = os.path.abspath('database/previously_seen_{}.txt'.format(today_date))
database = os.path.abspath('database/database_{}.tsv'.format(today_date))
with open(filename, 'w') as fp:
    pass
with open(database, 'w') as fp:
    pass


driver = webdriver.Chrome(r'main\chromedriver win 32.exe')

main_FinSME(data_set,today_date,filename,database)
main_businesswire(data_set,today_date,filename,database)
main_techcrunch(data_set,today_date,filename,database)
main_VCN(data_set,today_date,filename,database)
main_venturebeat(data_set,today_date,filename,database)
main_prnews(data_set,today_date,filename,database)
main_businessjournals(data_set,today_date,filename,database)
main_vcaonline(data_set,today_date,filename,database)
try:
    main_benzinga(data_set,today_date,filename,database)
except AttributeError:
    pass
main_globenewswire(data_set,today_date,filename,database)
main_pehub_new(driver,data_set,today_date,filename,database)
main_fortune(driver, today_date, database)
main_pep(driver, data_set, today_date, filename, database)
main_cb_news(driver, data_set, today_date, filename, database)
#main_axios(driver, data_set, today_date, filename, database)