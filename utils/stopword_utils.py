from nltk import download
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# Baixando o conjunto de palavras Stop Words da NLTK
download('stopwords')
palavras_irrelevantes = stopwords.words('portuguese')


# Definição da função para remover as StopWords e trasformar os comentarios em minusculo
def stopwords_removal(texto_lista, stopwords_lista=stopwords.words('portuguese')):
    '''
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto em que será removida as topwords
    stopwords_lista: list
        Lista contendo as stopword
    '''
    return [c.lower() for c in texto_lista.split() if c.lower() not in stopwords_lista]

# Criação de classe para remoção das stopwords
class RemoverStopwords(BaseEstimator, TransformerMixin):
    
    def __init__(self, text_stopwords):
        # text_stopwords lista de palavras StopWords
        self.text_stopwords = text_stopwords
    def fit(self, X, y=None):
        return self
    
    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        return [' '.join(stopwords_removal(comment, self.text_stopwords)) for comment in X]
