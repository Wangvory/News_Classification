import pandas as pd
import glob
import re
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
import pyLDAvis.gensim
import numpy as np
from gensim.models import CoherenceModel
import nltk
from sklearn.pipeline import Pipeline
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

all_files = [f for f in glob.glob("database/*.tsv") if re.match(r'database/database_\d+-\d+-\d+\.tsv',f)]
lst=[pd.read_table(f,header=None) for f in all_files]
frame = pd.concat(lst, ignore_index=True)

noise = frame[(frame.iloc[:,6]=='(No Keywords detect)')].iloc[:,0:6]
noise.columns=['time_scraped','time_published','source','Title','Link','text']
noise = noise.drop_duplicates(subset=['title'])
noise.to_csv('database/train_data/noise.csv')
# hand label some news
noise = pd.read_csv("database/train_data/noise.csv")
funding = pd.read_csv("database/train_data/funding.csv")
MA = pd.read_csv("database/train_data/M&A.csv")

MA['category'] = 'M&A'
funding['category'] = 'Funding'
noise['category'] = 'Non-Deal'
# Combine three data sources into a single dataframe
news_df = pd.concat([funding,MA,noise[['title','link','category']]],ignore_index=True)



#Text Preprocressing
stop_words = nltk.corpus.stopwords.words('english')
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()

def normalize_corpus(papers):
    norm_papers = []
    for paper in papers:
        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = list(filter(None, paper_tokens))
        if paper_tokens:
            norm_papers.append(paper_tokens)
    return norm_papers

tokenized_docs = normalize_corpus(news_df['title'])
print(len(tokenized_docs))

#Create Bigram Corpus
bigram = Phrases(tokenized_docs, min_count=3, threshold=5)
bigram_texts = [bigram[line] for line in tokenized_docs]
id2word = Dictionary(bigram_texts)
print('Total Vocabulary Size (Before):', len(id2word))

#convert to categories
corpus = [id2word.doc2bow(line) for line in bigram_texts]

def compute_coherence_values(corpus, id2word, k):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=len(bigram_texts),
                                           passes=10,
                                           per_word_topics=True)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=bigram_texts,
                                         dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()


cv_scores = []
for k in range(2, 9):
    score = compute_coherence_values(corpus=corpus, id2word=id2word, k=k)
    cv_scores.append(score)
    print(k, score)
import seaborn as sns
import matplotlib.pyplot as plt
optimal_k = np.array(cv_scores).argmax() + 2
sns.lineplot(x=np.arange(2, 9), y=cv_scores)
plt.show()

#LDA Model
lda_model = LdaMulticore(corpus=corpus,
                         id2word=id2word,
                         random_state=100,
                         num_topics=3,
                         passes=10,
                         chunksize=len(bigram_texts),
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)

plt.style.use('fivethirtyeight')
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=1200,
                  height=1200,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 3, figsize=(16,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16), y=1.05)
    plt.gca().axis('off')


plt.subplots_adjust(wspace=.3, hspace=.2)
plt.axis('off')
plt.margins(x=0, y=0)
#plt.tight_layout()
plt.show()

#select keywords
x = lda_model.show_topics(num_words=20)

twords = {}
for topic, word in x:
    twords[topic] = re.findall(r'\"(.*?)\"', word)

twords_df = pd.DataFrame(twords)
twords_df.columns = ['Topic ' + str(i) for i in range(optimal_k)]
#Build a feature dictionary
features = []
for words in twords.values():
    features.extend(words)
feature_dict = {word:i for i, word in enumerate(set(features))}
len(feature_dict)

#Build Text Classification Model
def featurize(texts, bigram_phrase, feature_dict):
    tokenized_docs = normalize_corpus(texts)
    bigram_texts = [bigram_phrase[line] for line in tokenized_docs]

    data = np.zeros((len(texts), len(feature_dict)))

    for i, text in enumerate(bigram_texts):
        for word in text:
            if word in feature_dict.keys():
                word_idx = feature_dict[word]
                data[i, word_idx] = 1
    return data

data = featurize(news_df['title'], bigram, feature_dict)
X = data
Y = news_df['category']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
pipeline = Pipeline([('clf', RandomForestClassifier())]) # Use Random Forest to identify topic
model = pipeline.fit(X_train, y_train)
ytest = np.array(y_test)
print(classification_report(ytest, model.predict(X_test)))

#Feature Importance
all_features = list(feature_dict.keys())
feature_importance = pd.DataFrame(model.steps[0][1].feature_importances_, index=all_features, columns=['feature_importance'])
feature_importance.sort_values('feature_importance').tail(30).plot.barh(figsize=[10,12])
plt.show()

#Apply the model to other unlabeld sources
df = frame.iloc[:,0:6]
df.columns = ['time_scraped','time_published','source','title','link','text']
df = df.drop_duplicates(subset=['title'])
df['all'] = df['title'] + ' ' + df['text']
df_new = df[df['all'].apply(lambda x: not isinstance(x, (float, int)))]

df_new['category_pred'] = model.predict(featurize(df_new['all'], bigram, feature_dict))
prob = pd.DataFrame(model.predict_proba(featurize(df_new['all'], bigram, feature_dict)), columns=model.steps[0][1].classes_).round(2)
df_new['Funding_prob'] = prob['Funding']
df_new['MA_prob'] = prob['M&A']
df_new['No_prob'] = prob['Non-Deal']

model.predict(featurize(df_new['title'], bigram, feature_dict))