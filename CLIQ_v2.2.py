# -*- coding: utf-8 -*-
"""CLIQ_v2.2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bf-bZ7-DqbRDm9d-CvJMENA20W2Cw7Ki
"""

# -*- coding: utf-8 -*-

"""CLIQ_v2.2.py

# CLIQ! Classificador Inteligente de Questões

por Rafael Ris-Ala
"""

"""# Componente 1: Libraries importation"""

# Instalação da versão mais recente do spaCy:
!pip install spacy==2.2.3
#!pip install spacy --upgrade

# Verifica a versão do spaCy:
import spacy
spacy.__version__

# Configuração dos pacotes em português
!python3 -m spacy download pt

# Importação das bibliotecas
import pandas as pd
import string
import spacy
import random
import numpy as np

"""# Componente 2: Dataset loading"""

# Autorizar o acesso ao seu diretório do Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Criar o dataframe q vamos usar no algoritmo
#base_dados = pd.read_csv('/content/drive/My Drive/Colab Notebooks/0. Mestrado/N161718treinoB.txt', delimiter = "\t", header=None, encoding="utf-8")
base_dados = pd.read_csv('/content/drive/My Drive/Colab Notebooks/0. Mestrado/Exemplo_2.2.csv', delimiter = ";", header=None, encoding="utf-8")

# Nomear as colunas do dataframe
base_dados.columns = ['questao', 'categoria']
#base_dados

# Visualizar o shape
#base_dados.shape

# Visualizar os 5 primeiros registros
#base_dados.head()

# Visualizar os 5 últimos registros
#base_dados.tail()

"""# Componente 3: Data treatment"""

"""## Limpeza dos textos"""

# Criar variável pontuações
pontuacoes = string.punctuation
#pontuacoes

# Importar o stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS

# Exibir a lista de stopwords
#print(stop_words)

# Exibir a quantidade de stopwords estão catalogadas
#len(stop_words)

# Criar o modelo em português
pln = spacy.load('pt')

# Verificar se o modelo foi carregado
pln

# Criar uma função para o pré-processamento do texto 
def preprocessamento(questao):
  # Converter para minúsculo:
  questao = questao.lower()
  documento = pln(questao)
  
  lista = []
  for token in documento:
    #lista.append(token.text) # Ao invés de pegar as palavras originais, pegaremos o lema! 
    # Lematização das palavras do documento:
    lista.append(token.lemma_)
  # Adicionar palavra na minha lista, se não estiver nas stopwords nem em pontuações:
  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
  # remoção de dígitos numéricos:
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

  return lista

# Testar a função criada:
#teste = preprocessamento('Estou aPrendendo 1 10 23 processamento de linguagem natural, Curso em Curitiba')
#teste

"""## Limpeza dos textos"""

# Exibir os 10 primeiros registros
#base_dados.head(10)

# Aplicação da função "preprocessamento"
base_dados['questao'] = base_dados['questao'].apply(preprocessamento)

# Exibir os 10 primeiros registros, agora pré-processados
#base_dados.head(10)

"""## Tratamento da classe"""

# Repare qual o formato da nossa base de dados "texto": 
# Vamos deixar nossa base de dados "texto" nesse formato lista com dicionário
base_dados_final = []
for questao, categoria in zip(base_dados['questao'], base_dados['categoria']):
  #print(texto, emocao)
  if categoria == 'facil':
    dic = ({'FACIL': True, 'MEDIA': False, 'DIFICIL': False})
  elif categoria == 'media':
    dic = ({'FACIL': False, 'MEDIA': True, 'DIFICIL': False})
  elif categoria == 'dificil':
    dic = ({'FACIL': False, 'MEDIA': False, 'DIFICIL': True})

  # Criar uma cópia dos dados formatados, senão fica apenas na memória 
  base_dados_final.append([questao, dic.copy()])

# Visualizar se foi criado corretamente
#len(base_dados_final)

# Exibindo um exemplo da frase e dicionário
#base_dados_final[0]

# Exibindo um exemplo da frase
#base_dados_final[0][0]

# Exibindo um exemplo do dicionário
#base_dados_final[0][1]

# Exibindo o tipo do dado
#type(base_dados_final[0][1])

# Exibir nossa base de dados:
base_dados_final

"""# Componente 4: Model training"""

# Agora que nossa base de dados está no formato requerido pela spaCy,
# vamos criar o modelo 

# Inicializar um novo modelo em branco:
modelo = spacy.blank('pt')
# Definir o tipo de caracterização do modelo, no caso, "textual"
categorias = modelo.create_pipe("textcat")
# Adicionar o rótulo "fácil"
categorias.add_label("FACIL")
# Adicionar o rótulo "média"
categorias.add_label("MEDIA")
# Adicionar o rótulo "difícil"
categorias.add_label("DIFICIL")
# Adicionar ao pipeine do spaCy...
modelo.add_pipe(categorias)
# Armazenar os resultados
historico = []
# Assim, nosso classificador estará nessa variável "modelo"

# Iniciar nosso treinamento
modelo.begin_training()
# Definir a quantidade de épocas q o algoritmo será executado, no caso, 1200:
for epoca in range(1200):
  # Misturar os dados, pq nossa base de dados estava ordenada (100 f, 100 m e 100 d)
  random.shuffle(base_dados_final)
  # Criar o dicionário (onde controlaremos o erro) 
  losses = {}
  # De quanto em quanto registros faremos a atualização dos pesos:
  # Como possuímos 196 registros, divididos por 30, teremos 6,5 ciclos, ou seja, vai pegando de 30 em 30 registros e submete à rede neural,
  # faz o cálculo do erro, depois faz o ajuste dos pesos. Até completar a 1º época. 
  # Depois repete isso até completar as 1200 épocas
  for batch in spacy.util.minibatch(base_dados_final, 30):
    # Vai pegar cada textos, tentar fazer um previsão e fazer o cálculo do erro
    textos = [modelo(texto) for texto, entities in batch]
    # Criaremos um variável com o texto e categoria
    annotations = [{'cats': entities} for texto, entities in batch]
    # Atualização (ajuste dos pesos), passando o texto, anotation e losses
    modelo.update(textos, annotations, losses=losses)
  # Observe q ele já vai exibindo o valor do erro. O objetivo é ir diminuindo-o de 100 em 100
  if epoca % 100 == 0:
    print(losses)
    historico.append(losses)

# Vamos salvar o modelo em disco temporário, ao fechar será perdido!! 
# Observe o diretório "modelo" criado na aba lateral do Colab contendo o "modelo"
# modelo.to_disk("modelo")
#modelo.to_disk("/content/drive/My Drive/Cursos - recursos/Modelo_2.2")

"""# Componente 5: Question classification"""

# Carregar o modelo que salvamos (para levar pronto para outra máquina e não termos que executar as etapas anteriores novamente):
# A partir do diretório "modelo" na aba lateral do Colab
#modelo = spacy.load("/content/drive/My Drive/Cursos - recursos/Modelo_2.2")
#modelo

# Criar um texto exemplo
#texto_exemplo = 'Qual associação em séries de pilhas fornece diferença de potencial, nas condições-padrão, suficiente para acender o LED azul?'
texto_exemplo = input("Insira uma questão de prova: ")
# Posso já passar o texto pré-processado
previsao = modelo(preprocessamento(texto_exemplo))
#previsao.cats

# Classificar a questão informada:
if(   (previsao.cats['DIFICIL'] >= previsao.cats['MEDIA']  ) 
  and (previsao.cats['DIFICIL'] >= previsao.cats['FACIL'] )):
  print('Essa questão é difícil!')
elif( (previsao.cats['MEDIA'] >= previsao.cats['DIFICIL']  ) 
  and (previsao.cats['MEDIA'] >= previsao.cats['FACIL'] )):
  print('Essa questão é média!')
elif( (previsao.cats['FACIL'] >= previsao.cats['DIFICIL']  ) 
  and (previsao.cats['FACIL'] >= previsao.cats['FACIL'] )):
  print('Essa questão é fácil!')