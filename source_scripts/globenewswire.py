import re
import os 
import sys
import os.path
import requests
from common_scripts import *
from datetime import datetime
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath("labeling"))
from labeling import *

sys.path.append(os.path.abspath("boto3"))
from split_db_sources import *

def extraction(key, url,data_set,seen,today,filename,database):
    
    soup = get_content(url)
    all_items = soup.find_all('item')
    all_articles = []
    article = {}
    
    for i in range(len(all_items)):
        
        article['title'] = all_items[i].find('title').get_text() 
        pubDate = all_items[i].find('pubDate').get_text()[:-4]
        pubDate = datetime.strptime(pubDate, '%a, %d %b %Y %H:%M')
        pubDate = pubDate.strftime("%m/%d/%Y, %H:%M:%S")
        article['pubDate']=pubDate
        article['link'] = all_items[i].find('link').get_text()
        article['description'] = cleanhtml(all_items[i].find('description').get_text())
        article['source'] = "Globenewswire -"+key
        if len(article['description']) >15 and 'Form 8.3' not in article['description'] and 'FORM 8.3' not in article['description']:
            
            all_articles.append(article)
        article = {}
    
    with open(database, "a", encoding='utf8') as rf:
        now = datetime.now()
        timenow = now.strftime("%m/%d/%Y, %H:%M:%S")
        
        with open(filename, 'a', encoding='utf8') as wf2:
            for i,article in enumerate(all_articles):
                if str(article['link']) not in data_set and str(article['link']) not in seen :
                    seen.add(str(article['link']))
                    article = label_creator(article)
                    nkw = '(No Keywords detect)'
                    if article['label_for_article_name'] == nkw and article['label_description'] == nkw:
                        pass
                    else:
                        arti = timenow+ '\t'+ article['pubDate'] + '\t' +article['source'] +'\t'+article['title']+"\t"+ str(article['link']) + \
                            '\t' + article['description'] + '\t' + article['label_for_article_name']  + '\t' + article['label_description']  + \
                            '\t' + article['Possible_ER_from_Article_Name'] +'\t'+ article["possible_ER_from_Comprehend"]
                        rf.write(arti+'\n')
                        if 'IPOs' in article['label_for_article_name']  or 'Bankruptcy' in article['label_for_article_name']:
                            create_file_bankruptcy_IPO(today_date, arti)
                        split_sources(arti)
                        wf2.write(timenow + '\t' + article['pubDate'] + '\t' +str(article['link'])+'\n') 
                        print(str(i)+ " "+arti[40:60]+'\n')


def main_globenewswire(data_set,today,filename,database):
    seen = set()
    urls ={'Mergers & Acquisitions':'https://www.globenewswire.com/RssFeed/subjectcode/27-Mergers%20And%20Acquisitions/feedTitle/GlobeNewswire%20-%20Mergers%20And%20Acquisitions',}
    
    for key in urls:
        extraction(key, urls[key],data_set,seen,today,filename,database)

