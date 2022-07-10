from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

def ngrams_count(texto_df, ngram_range, n=-1):
    '''
    Argumentos:
    ----------
    texto_df: pd.DataFrame
        Coluna contendo o texto a ser analisado
    ngram_range: Tuple(min_n, max_n)
        O limite inferior e superior do intervalo de valores n para diferentes n-grams de palavras ou carateres n-grams a serem extraídos
    n: Int, default=-1
        Número de elementos a serem selecionado no dataframe
    '''

    
    # Usando CountVectorizer para construir um pacote de palavras usando o o texto_lista fornecido
    vectorizer = CountVectorizer(ngram_range=ngram_range).fit(texto_df)
    bag_of_words = vectorizer.transform(texto_df)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]
    
    # Retornando o DataFrame com os N-Grams
    count_df = DataFrame(total_list, columns=['ngram', 'count'])
    return count_df
