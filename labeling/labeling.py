import nltk
import json
import boto3
import spacy
import numpy as np
import pandas as pd
import en_core_web_sm
from nltk import ne_chunk
from pprint import pprint
from spacy import displacy
from datetime import datetime
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer

nltk.download('stopwords')
nltk.download('punkt')
import en_core_web_sm
nlp = en_core_web_sm.load()
today_date = str(datetime.now())[:10]
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def get_comprehend_ER(example_text):

    text = example_text
    Entities = eval(json.dumps(comprehend.detect_entities(Text=text, LanguageCode='en'), sort_keys=True, indent=4))['Entities']
    ER = [('(\''+str(x['Text'])+'\',\''+str(x['Type'])+'\')') for x in Entities]
    ER = ''.join(ER)
    return ER  
 
def frequency_key_words(sentence):

    if type(sentence) != list:
        sentence_tokens = word_tokenize(sentence)
        stemmed_tokens = [ps.stem(x) for x in sentence_tokens if x not in stop_words]

    else:
        stemmed_tokens = [ps.stem(x) for x in sentence]
    # stemed keywords being looked for
    key_words = {'M&A': ('merger', 'merg', 'acquisit', 'acquir', 'buyout'),
                'Funding':('fund', 'rais', 'seri'),
                'Possible_Funding':('invest','score','round', 'pull', 'land', 'mln', 'garner', 'nab', 'ink', 'financ', 'pocket', 'million'),
                'Possible_M&A':('sell', 'ownership', 'sale', 'divest'),
                'IPOs': ('ipo', 'offer'),
                'Bankruptcy':('restructur', 'bankruptci','foreclosur','11','13')
                }

    checks = {'M&A':0,"Funding":0, 'Possible_M&A':0,"Possible_Funding":0, 'Bankruptcy':0, 'IPOs' :0 }
    for i,word in enumerate(stemmed_tokens):
        word = word.lower()
        if word in key_words['M&A']:
            checks['M&A'] += 1
        if word in key_words['Funding']:
            checks['Funding'] += 1
        if word in key_words['Possible_M&A']:
            checks['Possible_M&A'] += 1
        if word in key_words['Possible_Funding']:
            checks['Possible_Funding'] += 1
        if word in key_words['IPOs']:
            if word != 'offer':
                checks['IPOs'] += 1
            else:
                if i>0 and stemmed_tokens[i-1] == 'public':
                    checks['IPOs'] +=1
        if word in key_words['Bankruptcy']:
            if word not in set(['11','13']):
                checks['Bankruptcy'] += 1
            else:
                if i>0 and stemmed_tokens[i-1] =='chapter':
                    checks['Bankruptcy']+=1
    return checks 
  

def label_maker(d):
  label = ''
  lvalue = max(d.values())
  if lvalue == 0:
      return 'No Keywords detected'
  for i,val in d.items():
    if val == lvalue:
      label = label+i+' /'
  return label


def ER_using_spacy(example_text):
    doc = nlp(example_text)
    out = ''.join([str((X.text, X.label_)) for X in doc.ents])
    return(out)


def label_creator(article):
    article['title'] = article['title'].replace("\t","")
    if article['title']:
        identify  = frequency_key_words(article['title'])
        possible_ER = ER_using_spacy(article['title'])

        possible_ER_from_Comprehend = '-'
        set_identify =[]
        label_for_article_name = "("+label_maker(identify)[:-2]+")"
        label_description = 'Not Computed'

        if label_for_article_name == "(No Keywords detect)":
            if article['description']:
                identify2  = frequency_key_words(article['description'])
                label_description = "("+label_maker(identify2)[:-2]+")"

        article['label_for_article_name'] = label_for_article_name
        article["label_description"] = label_description  
        article["Possible_ER_from_Article_Name"] = possible_ER
        
        for x in identify.keys():
            if identify[x] == max(identify.values()) and article['label_for_article_name'] != '(No Keywords detect)':
                set_identify.append(x)
        # set_identify = set(set_identify)
        # if 'Bankruptcy' in set(set_identify) or 'IPOs' in set(set_identify):
        #     possible_ER_from_Comprehend = get_comprehend_ER(article['title'])
            
 
        article["possible_ER_from_Comprehend"] = possible_ER_from_Comprehend
        
    return article