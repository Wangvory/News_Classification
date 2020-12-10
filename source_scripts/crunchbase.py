import requests
from datetime import time, datetime
from bs4 import BeautifulSoup
from common_scripts import *

url = 'https://static.crunchbase.com/daily/content_share.html'



soup = get_content(url,None)
# -----------------------------------News feed-----------------------------------

all_items = soup.find('table').find('tbody').find_all('tr')[6].find('td')
length = len(all_items.find_all('h2'))
all_headers = all_items.find_all('h2')
all_links = [x.find('a', href=True)['href'] for x in all_items.find_all('h2')]
all_desc = [x.get_text() for x in all_items.find_all('p')]

all_articles = []
article = {}

date = soup.find('table').find('tbody').find_all('tr')[5].find('td').find('h3').get_text()

for idx in range(length):
    article['title'] = all_headers[idx].get_text()
    article['link'] = all_links[idx]
    article['description'] = all_desc[idx]
    # article['pubDate']  = date
    all_articles.append(article)
    article = {}

with open('cruchbase-News.tsv', "a") as rf:
    rf.write('\n')
    now = datetime.now()
    timenow = now.strftime("%m/%d/%Y, %H:%M:%S")
    
        # rf.write("Date_Collected"+'\t'+"Date_Published"+'\t'+"Source"+"\t" +"Article_Name"+"\t"+"Article_Link"+"\t"+"Description\n")
    
    for i,article in enumerate(all_articles):
        arti = article['title'] + '\t' + article['pubDate'] +"\t"+ str(article['link']) + '\t' + article['description'] 
        rf.write(arti+'\n')
        print(str(i)+ " "+arti[:20]+'\n')
                

# -------------------------------Recent Investment-------------------------------


all_investments = soup.find('table').find('tbody').find_all('tr')[22].find('tbody').find_all('tr')
companies_items = len(all_investments)
investment_data = []
invest = {}
date = soup.find('table').find('tbody').find_all('tr')[5].find('td').find('h3').get_text()
for i in range(0,companies_items,5) :
    company_info = all_investments[i].find_all('td')[1]
    company_name = company_info.find_all('p')[0].get_text().strip('\n')
    invest['company_name'] = company_name

    company_details =company_info.find_all('p')[1].find_all('a')[0].get_text() + '/'+ company_info.find_all('p')[1].find_all('a')[1].get_text()
    invest['company_details'] = company_details

    company_description = ("".join(all_investments[i+1].get_text().split('\n'))).strip()
    invest['company_description'] = company_description

    founder = str(all_investments[i+3].find('td').find_all('p')[1].find('a').get_text()) +' - ' + str(all_investments[i+3].find_all('td')[0].find_all('p')[1].find('a')['href'])
    invest['founder'] = founder

    investors = ''.join([(str(x.get_text())+', ') for x in all_investments[i+4].find('td').find_all('p')[1].find_all('a')]).strip().rstrip(',')
    invest['investors'] = investors

    investment_data.append(invest)
    invest = {}


# -------------------------------More Funding-------------------------------

more_funding =soup.find_all('table')[6].find('tbody').find_all('tr')
funding_table =[]
funding = {}
for tr in more_funding:
    tds = tr.find_all('td')
    funding['Company'] = tds[0].get_text().strip().strip('\n')
    funding['Amount/Round'] = ''.join([x.strip().strip('\n') for x in tds[1].get_text().split('\n')])
    funding['Lead_Investor'] = tds[2].get_text().strip().strip('\n')
    funding_table.append(funding)
    funding = {}

with open('cruchbase-Recent_Investments.tsv', "a") as rf:
        rf.write('\n')
        timenow = time.strftime('%X %x %Z')
        rf.write("Company_Name"+"\t" +"Amount/Round"+"\t"+"Date"+"\t"+"Founder"+"\t"+"Description"+"\t" +"Investors"+"\t"+timenow +"\n")
        for i,invest in enumerate(investment_data):
            entry = invest['company_name'] + '\t' + invest['company_details']+'\t'+date +"\t"+ invest['founder'] + '\t' +invest['company_description'] + '\t' + invest['investors']
            rf.write(entry+'\n')
            print(str(i)+ " "+entry[:20]+'\n')

with open('cruchbase-Recent_Funding.tsv', "a") as rf:
        rf.write('\n')
        rf.write("Company_Name"+"\t" +"Amount/Round"+"\t"+"Lead_Investor"+'\t'+timenow +"\n")
        for i,funing in enumerate(funding_table):
            entry = funing['Company'] + '\t' + funing['Amount/Round']+'\t'+funing['Lead_Investor'] 
            rf.write(entry+'\n')
            print(str(i)+ " "+entry[:20]+'\n')


# -------------------------------Recent Acquisitions-------------------------------

recent_acquisitions = soup.find('table').find('tbody').find_all('tr')[50]
acquisition = recent_acquisitions.find('table').find('tbody').find_all('tr')
companies_items = len(acquisition)
acq_data = []
acq = {}
date = soup.find('table').find('tbody').find_all('tr')[5].find('td').find('h3').get_text()

for i in range(companies_items) :
    acq['company_name'] = acquisition[i].find_all('td')[1].find_all('p')[0].find('a').get_text()
    acq['company_info'] = " ".join([x.strip() for x in acquisition[i].find_all('td')[1].find_all('p')[1].get_text().split('\n')])
    acq_data.append(acq)

# ------------------More Acquisitions ---------------------------------------
more_acq = recent_acquisitions.find_all('tbody')[1].find_all('tr')
more_acq_table = []
more_acq_data = {}
for x in more_acq:
    tds = x.find_all('td')
    more_acq_data['Company'] = tds[0].get_text()
    more_acq_data['Acquired_by'] = tds[1].get_text()
    more_acq_data['Amount'] = tds[2].get_text()
    more_acq_table.append(more_acq_data)


with open('cruchbase-Recent_Acquitions.tsv', "a") as rf:
        rf.write('\n')
        timenow = time.strftime('%X %x %Z')
        rf.write("Company_Name"+"\t" +"Company_info"+'\t'+timenow +"\n")
        for i,a in enumerate(acq_data):
            entry = a['company_name'] + '\t' + a['company_info'] 
            rf.write(entry+'\n')
            print(str(i)+ " "+entry[:20]+'\n')

with open('cruchbase-More_Acquisition.tsv', "a") as rf:
        timenow = time.strftime('%X %x %Z')
        rf.write('\n')
        rf.write("Company_Name"+"\t" +"Acquired_by"+"\t" +"Amount"+'\t'+timenow +"\n")
        for i,a in enumerate(more_acq_table):
            entry = a['Company'] + '\t' +a['Acquired_by']+ '\t' + a['Amount'] 
            rf.write(entry+'\n')
            print(str(i)+ " "+entry[:20]+'\n')
