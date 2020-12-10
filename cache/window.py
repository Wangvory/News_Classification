import os
import datetime
data_set = set()
from datetime import datetime, timedelta


def create_todays_prev_seen(date):
    if not os.path.exists('previously_seen'):
        os.makedirs('previously_seen')
    file = open('..\cache\previously_seen\previously_seen_{}.txt'.format(date), 'r') 
    data = file.read()

    data = data.split('\n')[:-1]
    all_rows = data

    data_set = []
    for x in data:
        date_link = x.split('\t')
        date_link[0]= date_link[0].strip("\"")
        data_set.append(date_link)
    file.close()

    # print(data_set)
# the path is relative to main 
    today_date = str(datetime.today())[:10]
    with open('..\cache\previously_seen\previously_seen_{}.txt'.format(today_date), 'w', encoding='utf8') as wf2:
        for i in range(len(data_set)):
            
            today = datetime.now()
            collected_date = datetime.strptime(data_set[i][0].strip("\""), '%m/%d/%Y, %H:%M:%S')
            published_date = datetime.strptime(data_set[i][1].strip("\""), '%m/%d/%Y, %H:%M:%S')

            collected_date = today - collected_date
            published_date = today - published_date

            collected_date = collected_date.total_seconds()
            published_date = published_date.total_seconds()

            days_since_collected  = int(divmod(collected_date, 86400)[0])
            days_since_published  = int(divmod(published_date, 86400)[0])

            if days_since_collected <5 or days_since_published > 6:
                #  the file is either a long term repetition or less than 5 days so old keep it 
                wf2.write(all_rows[i]+'\n')
            
