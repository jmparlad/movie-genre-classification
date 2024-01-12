
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

data_set = pd.read_csv('Movie_Data')

# Check if the data is balanced:
data_set['Main_Genre'].value_counts().plot(kind='bar')

# We need to balance the dataset. We won't take into account the genres
# with too few instances:

df_action = data_set[data_set['Main_Genre']=='Action']
df_comedy = data_set[data_set['Main_Genre']=='Comedy']
df_drama = data_set[data_set['Main_Genre']=='Drama']
df_crime = data_set[data_set['Main_Genre']=='Crime']
df_biography = data_set[data_set['Main_Genre']=='Biography']
df_adventure = data_set[data_set['Main_Genre']=='Adventure']
df_animation = data_set[data_set['Main_Genre']=='Animation']
df_horror = data_set[data_set['Main_Genre']=='Horror']

df_action_d = df_action.sample(100)
df_comedy_d = df_comedy.sample(100)#119
df_drama_d = df_drama.sample(100)
df_crime_d = df_crime.sample(100)
df_biography_d = df_biography.sample(100)
df_adventure_d = df_adventure.sample(100)
df_animation_d = df_animation.sample(100)
df_horror_d = df_horror.sample(100)

df_balanced = pd.concat([df_action_d, df_comedy_d,
                        df_drama_d, df_crime_d,
                        df_biography_d, df_adventure_d, df_animation_d,
                         df_horror_d])

df_balanced['Main_Genre'].value_counts()

# Train-test split:

g_labels = ['Action', 'Comedy', 'Drama', 'Crime',
            'Biography', 'Adventure', 'Animation',
            'Horror']

text = df_balanced['Synopsis'].tolist()
labels = df_balanced['Main_Genre'].apply(g_labels.index).tolist()

X_train, X_val, y_train, y_val = train_test_split(text,
                                                  labels,
                                                  stratify=labels,
                                                  test_size=.2)

# Prepare preprocessor function fot the bag-of-words models:
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

def preprocess(text):
    """Preprocess the given text by tokenising it, removing any stop words,
    replacing the remaining tokens with their lemma , and discarding
    all lemmas that contain non-alphabetical characters.
    """
    words = []
    for token in nlp(text):
        words.append(token)
    sw_list = [(token, token.is_stop) for token in words]
    rmvd_sw_list = []
    for tup in sw_list:
        if tup[1] == False:
            rmvd_sw_list.append(tup[0])
    lemma_list = [token.lemma_ for token in rmvd_sw_list]
    pp_text = []
    for token in lemma_list:
        if token.isalpha() == True:
            pp_text.append(token)

    return pp_text

# Define vectorizer
vectorizer = TfidfVectorizer(tokenizer = preprocess)

### Dummy classifier ###
pipe = Pipeline([('Vectorizer', vectorizer),('DC',DummyClassifier(strategy='stratified'))])
pipe.fit(X_train, y_train)
party_pred = pipe.predict(X_val)
print(classification_report(y_val, party_pred, target_names=g_labels))


### Naive Bayes Classifier ###
pipe2 = Pipeline([('Vectorizer', vectorizer),('MNB',MultinomialNB())])
pipe2.fit(X_train, y_train)
party_pred2 = pipe2.predict(X_val)
print(classification_report(y_val, party_pred2, target_names=g_labels))
cm = confusion_matrix(y_val, party_pred2)
reduced_g_labels = ['Actn', 'Com', 'Dram', 'Crime',
            'Biog', 'Adv', 'Anim', 'Horr']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=reduced_g_labels)
disp.plot()

### SVM ###
pipebaseSVM = Pipeline([('Vectorizer', vectorizer),('SVM',LinearSVC(dual= "auto"))])
parametersSVM = {'Vectorizer__binary':[True, False], 'Vectorizer__ngram_range':[(1,1),(1,2)],
                 'SVM__tol':[5e-5,1e-4,2e-4], 'SVM__C':[0.5, 1, 1.5], 'SVM__fit_intercept':[True, False]}
PipeGS2 = GridSearchCV(pipebaseSVM, parametersSVM, cv = 5)
PipeGS2.fit(X_train, y_train)
party_predGS2 = PipeGS2.predict(X_val)
print(classification_report(y_val, party_predGS2, target_names=g_labels))
cm = confusion_matrix(y_val, party_predGS2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=reduced_g_labels)
disp.plot()

### BERT classifier ###
from belt_nlp.bert_truncated import BertClassifierTruncated

MODEL_PARAMS = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 8,
    "num_labels": len(g_labels)
}
model = BertClassifierTruncated(**MODEL_PARAMS, device="cpu")
model.fit(X_train, y_train)
preds = model.predict_classes(X_val)
print(classification_report(y_val, preds,
                            target_names=g_labels))
cm = confusion_matrix(y_val, preds)
reduced_g_labels = g_labels = ['Actn', 'Com', 'Dram', 'Crime',
            'Biog', 'Adv', 'Anim', 'Horr']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=g_labels)
disp.plot()