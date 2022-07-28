from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import array, zeros, mean
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk import word_tokenize
import nltk

nltk.download('punkt')

def extract_features_from_corpus(texto_lista, vectorizer, df=False):
    '''
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto em que será normalizada
    vectorizer: object
        Engenharia utilizada para extração das features de texto
    '''

    # Extracting features
    corpus_features = vectorizer.fit_transform(texto_lista).toarray()
    features_names = vectorizer.get_feature_names_out()
    df_corpus_features = None
    if df:
        # Transforming into a dataframe to give interpetability to the process
        df_corpus_features = DataFrame(corpus_features, columns=features_names)
    
    return corpus_features, df_corpus_features

# Criação da classe para extrair features 
class ExtracaoFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, vectorizer):
        # vectorizer é a engenharia utilizada para extração das features de texto
        self.vectorizer = vectorizer
        
    def fit(self, X, y=None):
        return self
    
    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        return self.vectorizer.fit_transform(X).toarray()


'''
(SOUZA, 2019) SOUZA, ANTONIO ALEX DE. LUPPAR NEWS-REC: UM RECOMENDADOR INTELIGENTE DE NOTÍCIAS. 2019. 95 f. 
Dissertação (Mestrado Acadêmico em Computação) – Universidade Estadual do Ceará, , 2019. Disponível em: 
http://siduece.uece.br/siduece/trabalhoAcademicoPublico.jsf?id=93501
'''

# ReferÊncia (SOUZA, 2019)
class E2V_IDF(object):
    def __init__(self, word2vec):
        self.w2v = word2vec
        self.wIDF = None # IDF da palavra na colecao
        self.dimensao = 300
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        maximo_idf = max(tfidf.idf_) # Uma palavra que nunca foi vista (rara) então o IDF padrão é o máximo de idfs conhecidos (exemplo: 9.2525763918954524)
        self.wIDF = defaultdict(
            lambda: maximo_idf, 
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
        return self
    
    # Gera um vetor de 300 dimensões, para cada documento, com a média dos vetores (embeddings) dos termos * IDF, contidos no documento.
    def transform(self, X):
        return array([
                mean([self.w2v[word] * self.wIDF[word] for word in words if word in self.w2v] or [zeros(self.dimensao)], axis=0)
                for words in X
            ])
