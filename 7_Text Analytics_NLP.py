import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

text = """
Text Analytics is the process of analyzing text data using Natural Language Processing techniques.
It helps computers understand human language.
"""

tokens = word_tokenize(text)
tokens

pos = pos_tag(tokens)
pos

stop_words = set(stopwords.words('english'))
fillterd_words = []
for word in tokens:
    if word.lower() not in stop_words:
        fillterd_words.append(word)
fillterd_words

ps = PorterStemmer()
stemmed_words = []
for word in fillterd_words:
    stemmed_words.append(ps.stem(word))
stemmed_words

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in fillterd_words:
    lemmatized_words.append(lemmatizer.lemmatize(word))
lemmatized_words

documents = [
    "Text analytics is useful",
    "Natural language processing helps computers understand text",
    "Text analytics and NLP are important"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_matrix.toarray()

vectorizer.get_feature_names_out()