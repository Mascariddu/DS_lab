import csv

import re
import string

from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import treetaggerwrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class LemmaTokenizer(object):

    def __init__(self):
        self.stopwords = set(string.punctuation)
        self.stopwords.update(['per','che', 'del', 'con', 'sul', 'nel', 'dal', 'dell', 'alla',
                               'avere', 'essere', 'il', 'ci', 'di', 'da', 'la', 'lo', 'un', 'le',
                               'stare', 'anche', 'quale', 'fare', 'una', 'uno', 'dalla', 'dal', 'quest',
                               'quel', 'quest', 'quell', 'quel', 'fra', 'tra', 'sul', 'sull', 'volere',
                               'noi', 'lor', 'loro', 'dagl', 'agl', 'coi', 'sui', 'dei',
                               'so', 'sa', 'al', 'ok', 'in', 'fa', 'nè', 'se', 'anch', 'gli', 'gl',
                               'voler', 'noi', 'lor', 'mi', 'si', 'li', 'nell', 'dall', 'questo', 'quello',
                               'quella', 'quelle', 'questa', 'queste', 'della', 'dello', 'delle', 'alle',
                               'sulla', 'sulle', 'sugli', 'dagli', 'agli', 'degli', 'nelle', 'nello', 'nella',
                               'negli'
                               ])

    def __call__(self, documents):
        lemmas = []
        for t in word_tokenize(documents):
            t = t.strip()
            lemma = t
            if not re.findall('\W|(_)', lemma) and len(lemma) > 1:
                lemmas.append(lemma)
        return lemmas


# Pre - processing
files = []
labels = []
evaluation_set = []

print("Loading data")
df = pd.read_csv("/Users/andreamascarello/Desktop/dataset_winter_2020/development.csv", encoding='utf-8')
emoticons =  ('😇','😊','❤️','😘','💞','💖','🤗','💕','👏','🎉','👍',
              '😂','😡','😠','😭','🤦‍','🤷🏼‍','😞','👎','😱','😓','🔝')

df.text = df.text.replace('[^ a-zA-Zà-ú'
                            '\😇\😊\❤️\😘\💞\💖\🤗\💕\👏\🎉\👍'
                            '\😂\😡\😠\😭\🤦‍\🤷🏼‍\😞\😱\👎\😓\🔝]', " ",regex=True)
for word in emoticons:
    df.text = df.text.replace(word, " "+word+" ",regex=True)

df.text = df.text.replace('\s+', ' ',regex=True)
df.text = df.text.replace('^ ', '', regex=True)
df.text = df.text.replace(' $', '', regex=True)
df.text = df.text.apply(lambda x: x.lower())
df.text = df.text.replace('^', ' ', regex=True)
df.text = df.text.replace('$', ' ', regex=True)


eval = []
df2 = pd.read_csv("/Users/andreamascarello/Desktop/dataset_winter_2020/evaluation.csv", encoding='utf-8')

emoticons =  ('😇','😊','❤️','😘','💞','💖','🤗','💕','👏','🎉','👍',
              '😂','😡','😠','😭','🤦‍','🤷🏼‍','😞','👎','😱','😓','🔝')

df2.text = df2.text.replace('[^ a-zA-Zà-ú'
                            '\😇\😊\❤️\😘\💞\💖\🤗\💕\👏\🎉\👍'
                            '\😂\😡\😠\😭\🤦‍\🤷🏼‍\😞\😱\👎\😓\🔝]', " ",regex=True)
for word in emoticons:
    df2.text = df2.text.replace(word, " "+word+" ",regex=True)

df2.text = df2.text.replace('\s+', ' ',regex=True)
df2.text = df2.text.replace('^ ', '', regex=True)
df2.text = df2.text.replace(' $', '', regex=True)
df2.text = df2.text.apply(lambda x: x.lower())
df2.text = df2.text.replace('^', ' ', regex=True)
df2.text = df2.text.replace('$', ' ', regex=True)

df2.text = df2.text.apply(lambda x: x.strip())

files = df['text']
labels = df['class']
evaluation_set = df2['text']

files_new = []
tagger = treetaggerwrapper.TreeTagger(TAGLANG='it')
for idx, row in enumerate(files):
    tags = tagger.tag_text(row)
    tags2 = treetaggerwrapper.make_tags(tags)
    tags2 = np.array(tags2)
    out = ''
    for el in tags2:
            if el[2] == '@card@':
                out += ' ' + el[0]
            else:
                out += ' ' + el[2]
    files_new.append(out)
    with open('d_clean.csv', 'a') as fd:
        fd.write(out + ',' + labels[idx] + '\n')

files = files_new
files_new = []
for row in evaluation_set:
    tags = tagger.tag_text(row)
    tags2 = treetaggerwrapper.make_tags(tags)
    tags2 = np.array(tags2)
    out = ''
    for el in tags2:
            if el[2] == '@card@':
                out += ' ' + el[0]
            else:
                out += ' ' + el[2]
    files_new.append(out)
    with open('ev_clean.csv', 'a') as fd:
        fd.write(out + '\n')

evaluation_set = files_new
print('write finished')

pos = []
pos_labels = []
neg = []
neg_labels = []

files = []
labels = []
evaluation_set = []

with open("d_clean.csv") as f:
    for row in csv.reader(f):
        files.append(row[0])
        if row[1] == 'pos':
            pos.append(row[0])
        else:
            neg.append(row[0])
        labels.append(row[1])

with open("ev_clean.csv") as f:
    for row in csv.reader(f):
        evaluation_set.append(row[0])

lab = 'Positive', 'Negative'
sizes = [(len(pos)/len(files)) * 100, (len(neg) / len(files)) * 100]
explode = (0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=lab, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of labels')

plt.show()

sns.set(style="darkgrid")
ax = sns.countplot(data=df, x='class', palette=["#7fcdbb","#edf8b1"])
plt.title('Count of labels')
plt.show()

print("Vectorizing")
lemmaTokenizer = LemmaTokenizer()
vectorizer = TfidfVectorizer(tokenizer=lemmaTokenizer, min_df=2, max_df=1.0, max_features=13000, ngram_range=(1, 3),
                             stop_words=lemmaTokenizer.stopwords)
vectorizer2 = TfidfVectorizer(tokenizer=lemmaTokenizer, min_df=2, max_df=1.0, ngram_range=(1, 3), 
                              stop_words=lemmaTokenizer.stopwords)

words = vectorizer.fit_transform(neg)
features = np.array(vectorizer.get_feature_names())
temp_df = pd.DataFrame(words.todense(), columns=features)
p = temp_df.sum().sort_values()
p.tail(20).plot(kind='barh', color='turquoise')
plt.title('Most frequent tokens in negative reviews')
plt.show()

words = vectorizer.fit_transform(pos)
features = np.array(vectorizer.get_feature_names())
temp_df = pd.DataFrame(words.todense(), columns=features)
p = temp_df.sum().sort_values()
p.tail(20).plot(kind='barh', color='gold')
plt.title('Most frequent tokens in positive reviews')
plt.show()

tfidf_X = vectorizer2.fit_transform(files)

# Classification
print("Classification")
passive = PassiveAggressiveClassifier(random_state=98, fit_intercept=True, class_weight={'pos': 0.6792794046045768,
                                                                                         'neg': 0.3207205953954232})
# Train the model on training data
passive.fit(tfidf_X, labels)

# Prediction
tfidf_eval = vectorizer2.transform(evaluation_set)

predictions = passive.predict(tfidf_eval)

print(predictions)

pos = []
neg = []

for val in predictions:
    if val == 'pos':
        pos.append(val)
    else:
        neg.append(val)

lab = 'Positive', 'Negative'
sizes = [(len(pos)/len(files)) * 100, (len(neg) / len(files)) * 100]
explode = (0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=lab, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of labels')

plt.show()



def dump_to_file(filename, assignments):
    with open(filename, mode="w", newline="") as csvfile:
        # Headers
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ids, cluster in enumerate(assignments):
            writer.writerow({'Id': str(ids), 'Predicted': str(cluster)})


dump_to_file("sample_submission.csv", predictions)
print("Computed finished")


def generate_wordclouds(tfidf, predicted, model, vectorizer):

    classes = set(predicted)
    word_positions = {v: k for k, v in vectorizer.vocabulary_.items()}
    top_count = 100
    for cluster_id in classes:
        # compute the total tfidf for each term in the cluster
        X = tfidf[predicted == cluster_id]
        x_sum = np.sum(X, axis=0) # numpy.matrix
        x_sum = np.asarray(x_sum).reshape(-1)

        coef = np.asarray(model.coef_).reshape(-1)

        if cluster_id == 'pos':
            print('pos')
            top_indices = coef.argsort()[-top_count:]
        else:
            print('neg')
            top_indices = coef.argsort()[:top_count]

        term_weights = {word_positions[idx]: x_sum[idx] for idx in top_indices}
        term_weights = sorted(term_weights.items(), key=lambda x: x[1])
        term_weights = dict(term_weights)
        print(term_weights)
        wc = WordCloud(width=1200, height=800, background_color="white")
        wordcloud = wc.generate_from_frequencies(term_weights)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.imshow(wordcloud, interpolation='bilinear')
        4
        ax.axis("off")
        fig.suptitle(f"Cluster {cluster_id}")
        plt.show()


generate_wordclouds(tfidf_eval, predictions, passive, vectorizer2)

print('Tuning')


def tuning():
    print('Tuning started..')
    kfold = StratifiedKFold(n_splits=5, random_state=98, shuffle=True)
    f = 0
    best = 0
    for train_id, test_id in kfold.split(files, labels):
        f += 1
        print(' - Fold', f)
        X_train = np.array(files)[train_id]
        y_train = np.array(labels)[train_id]
        X_test = np.array(files)[test_id]
        y_test = np.array(labels)[test_id]

        pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer=lemmaTokenizer, stop_words=lemmaTokenizer.stopwords)),
                             ('passive', PassiveAggressiveClassifier(random_state=98))])

        params = {
            'vect__min_df': [2, 5, 7, 10],
            'vect__max_df': [0.7, 0.8, 0.9, 1.0],
            'vect__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5)],
            'passive__fit_intercept': [True, False],
            'passive__class_weight': [{'pos': 1, 'neg': 0.75}, {'pos': 1, 'neg': 0.8}, {'pos': 0.6792794046045768,
                                                                                        'neg': 0.3207205953954232}]
        }

        grid_search = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        predicted = grid_search.predict(X_test)

        score = f1_score(y_test, predicted, average='weighted')
        print('F1 score: ', score)
        print('Parameters', grid_search.best_params_)

        if score > best:
            best = score
            best_config = grid_search.best_params_

    return best_config


print(tuning())


