import matplotlib.pyplot as plt
from wordcloud import WordCloud

def nuvem_palavras(dataset, coluna_texto, n_score):
    '''
    dataset: Dataframe (Pandas)
        Conjunto de dados 
    coluna_texto: String
        Nome da coluna que possui os textos a serem claficados
    n_score: Integer
        Valor do Score que se dejesa plotar a nuvem de palavras (WordCloud)
    '''
    score = dataset[dataset['sentiment_label'] == n_score]
    palavras = ' '.join([texto for texto in score[coluna_texto]])
    # gerando a wordcloud
    nuvem_palavras = WordCloud(width=800,\
                                height=500, 
                                    max_font_size=110,\
                                        collocations=False).generate(palavras)

    # trasformando o objeto gerado em uma imagem
    plt.figure(figsize=(14, 8))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    