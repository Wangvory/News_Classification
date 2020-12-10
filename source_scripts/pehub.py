DEPRICATED

import os
import re
import sys
import requests
from common_scripts import *
from datetime import datetime
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath("..\labeling"))
from labeling import *

main_url = 'https://www.pehub.com/news-briefs/page/1/'

urls = {'VC_DEALS': 'https://pehub.com/category/vc-deals/feed',
'PE_DEALS': 'https://pehub.com/category/buyout-deals/feed',
'IPOS': 'https://pehub.com/category/pe-backed-ipos/feed',
'M&A': 'https://pehub.com/category/pe-backed-ma/feed',
'FIRMS&FUNDS': 'https://pehub.com/category/firms-funds/feed'}



def get_feed(name, url,data_set,seen,today,filename,database):
     
    soup = get_content(url,None)
    all_items = soup.find_all('item')
    print(soup)
    all_articles = []
    article = {}

    for idx in range(len(all_items)):
        print(all_items[idx])
        article['title'] = all_items[idx].find('title').get_text()
        article['link'] = all_items[idx].find('link').get_text().strip('\t')
        pubDate = all_items[idx].find('pubDate').get_text()[:-6]
        pubDate = datetime.strptime(pubDate, '%a, %d %b %Y %H:%M:%S')
        pubDate = pubDate.strftime("%m/%d/%Y, %H:%M:%S")
        article['pubDate']  = pubDate
        article['description'] = cleanhtml(all_items[idx].find('description').get_text().strip('\n').strip('\t'))
        article['source'] = 'PEHub-'+name
        all_articles.append(article)
        article = {}


    with open(database, "a", encoding='utf-8') as rf:
        now = datetime.now()
        timenow = now.strftime("%m/%d/%Y, %H:%M:%S")

        with open(filename, 'a', encoding='utf8') as wf2:
             with open('..\cache\duplicates3.tsv', 'a', encoding='utf8') as wf_duplicates:
                for i,article in enumerate(all_articles):
                    # if str(article['link']) not in data_set and str(article['link']) not in seen  and check_publish_date(timenow, article['pubDate']):
                    seen.add(str(article['link']))
                    article = label_creator(article)
                    arti = timenow+ '\t'+ article['pubDate'] + '\t' +article['source'] +'\t'+article['title']+"\t"+ str(article['link']) + '\t' + article['description'] + '\t' + article['label_for_article_name']  + '\t' + article['label_description']  + '\t' + article['Possible_ER_from_Article_Name']
                    print(arti)
                    print('\n')
                    #     rf.write(arti+'\n')
                    #     wf2.write(timenow + '\t' + article['pubDate'] + '\t' +str(article['link'])+'\n')
                    #     print(str(i)+ " "+arti[40:60]+'\n')
                    # else:
                    #     arti =  timenow+'\t'+article['source'] +'\t'+article['title']+"\t"+ str(article['link']) + '\t' + article['description'] + '\t' + "1"
                    #     wf_duplicates.write(arti+'\n')


def main_pehub(data_set,today,filename,database):

    seen = set()
    for name in urls :
        get_feed(name,urls[name],data_set,seen,today,filename,database)
