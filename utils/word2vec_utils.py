from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from numpy import array, mean, zeros

'''
(SOUZA, 2019) SOUZA, ANTONIO ALEX DE. LUPPAR NEWS-REC: UM RECOMENDADOR INTELIGENTE DE NOTÍCIAS. 2019. 
95 f. Dissertação (Mestrado Acadêmico em Computação) – Universidade Estadual do Ceará, , 2019. Disponível 
em: http://siduece.uece.br/siduece/trabalhoAcademicoPublico.jsf?id=93501 Acesso em: 27 de fevereiro de 2020
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