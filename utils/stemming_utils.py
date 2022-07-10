from nltk import download
from nltk import RSLPStemmer
from sklearn.base import BaseEstimator, TransformerMixin

# Realizando download do stemming para a linguagem Pt-br
download('rslp')

def stemming_process(texto_lista, stemmer=RSLPStemmer()):
    '''
    Argumentos:
    ----------
    texto_lista: list
        Lista contentdo o texto em que será normalizada
    stemmer: class, default: RSLPStemmer()
        Tipo de Stemmer que será aplicado ao texto
    '''
    
    return [stemmer.stem(c) for c in texto_lista.split()]


# Criação da classe para aplicar o processo de stemming
class ProcessoStemming(BaseEstimator, TransformerMixin):
    
    def __init__(self, stemmer):
        # Stemming é a técnica que transforma as flexões de uma palavra em um núcleo comum (tronco)
        self.stemmer = stemmer
    
    def fit(self, X, y=None):
        return self

    # Sobrescrevendo o método transform da classe pai TransformerMixin
    def transform(self, X, y=None):
        return [' '.join(stemming_process(comment, self.stemmer)) for comment in X]