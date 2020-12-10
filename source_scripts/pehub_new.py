import os
import re
import sys
import requests
from common_scripts import *
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

sys.path.append(os.path.abspath("labeling"))
from labeling import *

sys.path.append(os.path.abspath("boto3"))
from split_db_sources import *


def pehub_page_numbered(pg, driver, data_set, today, filename,database):
    if pg ==1 :
        driver.get('https://www.pehub.com/news-briefs/')
    else:
        driver.get('https://www.pehub.com/news-briefs/page/'+str(pg)+'/')
    driver.implicitly_wait(150)
    html = driver.page_source
    html = BeautifulSoup(html,'lxml')
    soup = html.find('div', {'class':'td-ss-main-content'})
    all_items = soup.find_all('div', {'class':'td_module_16 td_module_wrap td-animation-stack'})
    all_articles = []
    article = {}

    for idx in range(len(all_items)):
    #     print(all_items[idx])
        article['title'] = all_items[idx].find('h2',{'class':'entry-title td-module-title'}).get_text()
        article['link'] = all_items[idx].find('h2',{'class':'entry-title td-module-title'}).find('a')['href'].strip('\t')
        pubDate = all_items[idx].find('span',{'class':'td-post-date'}).find('time')['datetime'][:-6]
        pubDate = datetime.strptime(pubDate, '%Y-%m-%dT%H:%M:%S')
        pubDate = pubDate.strftime("%m/%d/%Y, %H:%M:%S")
        article['pubDate']  = pubDate
        description = cleanhtml(all_items[idx].find('div', {'class':'td-excerpt'}).get_text())
        article['description'] = description.strip('\t')
        article['source'] = 'PEHub'
        all_articles.append(article)
        article = {}

    with open(database, "a", encoding='utf-8') as rf:
        now = datetime.now()
        timenow = now.strftime("%m/%d/%Y, %H:%M:%S")

        with open(filename, 'a', encoding='utf8') as wf2:
            
            for i,article in enumerate(all_articles):
                
                if str(article['link']) not in data_set and str(article['link'])  :
                    article = label_creator(article)
                    nkw = '(No Keywords detect)'
                    if article['label_for_article_name'] == nkw and article['label_description'] == nkw:
                        pass
                    else:
                        arti = timenow+ '\t'+ article['pubDate'] + '\t' +article['source'] +'\t'+article['title']+"\t"+ str(article['link']) + \
                        '\t' + article['description'] + '\t' + article['label_for_article_name']  + '\t' + article['label_description']  + '\t' \
                        + article['Possible_ER_from_Article_Name'] +'\t'+ article["possible_ER_from_Comprehend"]
                        rf.write(arti+'\n')
                        if 'IPOs' in article['label_for_article_name']  or 'Bankruptcy' in article['label_for_article_name']:
                            create_file_bankruptcy_IPO(today_date, arti)
                        split_sources(arti)
                        wf2.write(timenow + '\t' + article['pubDate'] + '\t' +str(article['link'])+'\n') 
                        print(str(i)+ " "+arti[40:60]+'\n')
                
def main_pehub_new(driver,data_set,today,filename,database):
    
    for pg in range(1,6):
        pehub_page_numbered(pg, driver, data_set,today,filename,database)