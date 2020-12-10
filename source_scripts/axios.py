import re
import os
import sys
import time 
import requests
import pandas as pd
from lxml import html
from common_scripts import *
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait

sys.path.append(os.path.abspath("labeling"))
from labeling import *

sys.path.append(os.path.abspath("boto3"))
from boto3.split_db_sources import *

driver = webdriver.Chrome

def main_axios(driver,data_set,today_date,filename,database):
     
    driver.get('https://www.axios.com/newsletters/axios-pro-rata')
    driver.implicitly_wait(100)
    html = driver.page_source
    html = BeautifulSoup(html,'lxml')
    now = datetime.now()    
    timenow = now.strftime("%m/%d/%Y, %H:%M:%S")
    d = {}
    soup = html.find("div",{"id":"maincontent"})
    
    with open(database, "a", encoding='utf8') as rf:
            with open(filename, 'a', encoding='utf8') as wf2:                    
                for i in range(2,9):
                    deals = soup.findChildren("div" , recursive=False)[i].findChildren('div', recursive='False')[1]
                    deals = deals.find('div',{'class':'StoryText__StyledStoryText-b0w77w-0 ioucAl story-text gtm-story-text'}).findChildren('p')
                    for deal in deals:
                        d['title'] = ", ".join([x.get_text() for x in deal.find_all('strong')])
                        d['title'] = d['title'].replace("•,", "")
                        description = deal.get_text()
                        numbers = set([x for x in range(48,58,1)])
                        caps = set([x for x in range(65,91,1)])
                        lower = set([x for x in range(97,123,1)])
                        special = set([24,21,20,22,26,28,29,40])
                        allowed = numbers | caps | special | lower
                        while len(description) >0 and ord(description[0]) not in allowed:
                            description = description[1:]
                        d['description'] = description
                        d['link'] = deal.find('a')
                        if d['link']:
                            d['link'] = deal.find('a').get_text()          
                        d['pubDate'] = timenow
                        d['source'] = 'Axios'
                        article = label_creator(d)
                        nkw = '(No Keywords detect)'
                        if article['label_for_article_name'] == nkw and article['label_description'] == nkw:
                            pass
                        else:
                            arti = timenow+ '\t'+ article['pubDate'] + '\t' +article['source'] +'\t'+article['title']+"\t"+ str(article['link']) + \
                            '\t' + article['description'] + '\t' + article['label_for_article_name']  + '\t' + article['label_description']  + '\t' \
                            + article['Possible_ER_from_Article_Name'] + '\t'+ article["possible_ER_from_Comprehend"]
                            rf.write(arti+'\n')
                            if 'IPOs' in article['label_for_article_name']  or 'Bankruptcy' in article['label_for_article_name']:
                                create_file_bankruptcy_IPO(today_date, arti)
                            rf.write(arti+'\n')
                            split_sources(arti)
                            wf2.write(timenow + '\t' + d['pubDate'] + '\t' +str(d['link'])+'\n')  
                        d ={}
