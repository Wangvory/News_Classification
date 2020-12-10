
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
    
    soup = get_content(url,None)
    all_items = soup.find_all('item')
    all_articles = []
    article = {}
    
    for i in range(len(all_items)):
        article['title'] = all_items[i].find('title').get_text() 
        pubDate = all_items[i].find('pubDate').get_text()[:-6]
        pubDate = datetime.strptime(pubDate, '%a, %d %b %Y %H:%M:%S')
        pubDate = pubDate.strftime("%m/%d/%Y, %H:%M:%S")
        article['pubDate']=pubDate
        article['link'] = all_items[i].find('link').get_text()
        article['description'] = all_items[i].find('description').get_text()[3:-4]
        article['source'] = "PRNewswire -"+key
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
                        '\t' + article['description'] + '\t' + article['label_for_article_name']  + '\t' + article['label_description']  + '\t' \
                        + article['Possible_ER_from_Article_Name'] +'\t'+ article["possible_ER_from_Comprehend"]
                        
                        rf.write(arti+'\n')
                        if 'IPOs' in article['label_for_article_name']  or 'Bankruptcy' in article['label_for_article_name']:
                            create_file_bankruptcy_IPO(today_date, arti)
                        split_sources(arti)
                        wf2.write(timenow + '\t' + article['pubDate'] + '\t' +str(article['link'])+'\n') 
                        print(str(i)+ " "+arti[40:60]+'\n')
                    
# cases where prnews.py is run as standalone, otherwise data_set comes from main.py                       # 
data_set = set()
if os.path.isfile('..\cache\previously_seen3.txt'):
    file = open('..\cache\previously_seen3.txt', 'r') 
    data = file.read()
    data = data.split('\n')[:-1]
    data_set = set(data)
    file.close()
    

def main_prnews(data_set,today,filename,database):
    seen = set()
    urls ={'All News Releases':'https://www.prnewswire.com/rss/news-releases-list.rss',
    'Auto & Transportation':'https://www.prnewswire.com/rss/automotive-transportation-latest-news.rss',
    'Business Technology':'https://www.prnewswire.com/rss/business-technology-latest-news/business-technology-latest-news-list.rss',
    'Business Technology':'https://www.prnewswire.com/rss/business-technology-latest-news/business-technology-latest-news-list.rss',
    'Consumer Products & Retail': 'https://www.prnewswire.com/rss/consumer-products-retail-latest-news/consumer-products-retail-latest-news-list.rss',
    'Consumer Technology': 'https://www.prnewswire.com/rss/consumer-technology-latest-news/consumer-technology-latest-news-list.rss',
    'Energy': 'https://www.prnewswire.com/rss/energy-latest-news/energy-latest-news-list.rss',
    'Entertain­ment & Media': 'https://www.prnewswire.com/rss/entertainment-media-latest-news/entertainment-media-latest-news-list.rss',                   
    'Environ­ment':'https://www.prnewswire.com/rss/environment-latest-news/environment-latest-news-list.rss',
    'Financial Services & Investing':'https://www.prnewswire.com/rss/financial-services-latest-news.rss',
    'General Business':'https://www.prnewswire.com/rss/general-business-latest-news/general-business-latest-news-list.rss',
    'Heavy Industry & Manufacturing':'https://www.prnewswire.com/rss/heavy-industry-manufacturing-latest-news/heavy-industry-manufacturing-latest-news-list.rss',
    'Health':'https://www.prnewswire.com/rss/health-latest-news/health-latest-news-list.rss',
    'People & Culture':'https://www.prnewswire.com/rss/multicultural-latest-news/multicultural-latest-news-list.rss',
    'Policy & Public Interest':'https://www.prnewswire.com/rss/policy-public-interest-latest-news/policy-public-interest-latest-news-list.rss',
    'Telecomm­unications':'https://www.prnewswire.com/rss/telecommunications-latest-news/telecommunications-latest-news-list.rss',
    'Sports':'https://www.prnewswire.com/rss/sports-latest-news/sports-latest-news-list.rss',
    'Travel':'https://www.prnewswire.com/rss/travel-latest-news/travel-latest-news-list.rss'}
    
    for key in urls:
        extraction(key, urls[key],data_set,seen,today,filename,database)


# if not os.path.isfile('..\cache\duplicates3.tsv'):
#
#     file = open('..\cache\duplicates3.tsv', 'w')
#     file.write("Date_Collected\tSource\tArticle_Name\tArticle_Link\tDescription\tCount\n")
#     file.close()
    
# main_prnews(data_set)