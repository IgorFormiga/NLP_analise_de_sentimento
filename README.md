# **Projeto: Análise de Sentimentos em Português utilizando Machine Learning**

### **Objetivo:** Criação de um modelo de Machine Learning de análise de sentimento (classificação) utilizado o processamento de linguagem natural. O modelo deve ser capaz de prever automaticamente os sentimentos dos reviews, sem a necessidade de serem rotuladas por humanos.

### **Linguagem de programação:** Python

### **Etapas:**

- Compreensão da base de dados;
- Preparação dos Dados (Pepiline) e Embeddings (CountVectorize | TF-IDF)
- Criação dos Modelos 

### **Data Science Lifecycle:**
- 1º ciclo (analise_sentimento.ipynb): Desenvolvimento de um modelo de análise de sentimento que irá irá classificar os comentários em três classes: "Positivo", "Negativo" e "Neutro". 
- 2º ciclo (2_analise_sentimento.ipynb): Troca do processo de stemming por lematização, remoção de ~10% de tokens que mais aparecem e que menos aparecem e incluir filtro Pos Tag.
- 3º ciclo (3_analise_sentimento.ipynb): Classificar os comentários em duas classes: "Other" e "Negativo", em vez de três classes como definido no primeiro objetivo do projeto. (*em desenvolvimento*)

### **Base de dados:** ([link](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce))
O projeto foi desenvolvido utilizado um conjunto de dados público de e-commerce brasileiro de pedidos feitos na Olist Store, a maior loja de departamentos dos marketplaces brasileiros. O dataset possui informações de 100 mil pedidos de 2016 a 2018 feitos em vários marketplaces no Brasil. 

A base de dados de reviews foi construída a partir compras de produtos por clientes na Olist Store. Assim que o cliente recebe o produto, ou vence a data prevista de entrega, o cliente recebe uma pesquisa de satisfação por e-mail onde pode dar uma nota da experiência de compra e anotar alguns comentários.

### **Requirements:**
- requirements-nb: biblioecas utilizadas para execução dos arquivos do tipo IPython Notebook (*.ipynb*)
- requirements: biblioecas utilizadas para execução dos arquivos do tipo Python (*.py*)