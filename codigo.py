# -*- coding: utf-8 -*-
"""Modelo CLIQ.ipynb

# Modelo CLIQ

por Rafael Ris-Ala

# Etapa 1: Importação e instalação das bibliotecas
"""

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
import seaborn as sns
import numpy as np

"""# Etapa 2: Carregamento da base de dados (+ Mary)"""

#carregamento de arquivos (N161718treinoB.txt)
#from google.colab import files
#uploaded = files.upload()
##Ou
##Autorizo acesso ao meu diretório do Google Drive
from google.colab import drive
drive.mount('/content/drive')

#criação do dataframe q vamos usar no algoritmo
#import pandas as pd
#base_dados = pd.read_csv('N161718treinoB.txt', delimiter = "\t", header=None, encoding="utf-8")
##ou
import pandas as pd
base_dados = pd.read_csv('/content/drive/My Drive/Colab Notebooks/0. Mestrado/N161718treinoB.txt', delimiter = "\t", header=None, encoding="utf-8")

#nomeação das colunas do dataframe
base_dados.columns = ['questao', 'categoria']
base_dados

# Visualizar o shape
base_dados.shape

# Visualizar os 5 primieros registros
base_dados.head()

# Visualizar os 5 últimos registros
base_dados.tail()

# Visualizar dos registros em gráfico
sns.countplot(base_dados['categoria'], label = 'Contagem');

"""# Etapa 3: Função para pré-processamento dos textos


*   Como remoção de: pontuação, stopwords,


"""

# Criar variável pontuações
pontuacoes = string.punctuation
pontuacoes

# Importar o stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS

# Exibir a lista de stopwords
print(stop_words)

# Exibir a quantidade de stopwords estão catalogadas
len(stop_words)

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
teste = preprocessamento('Estou aPrendendo 1 10 23 processamento de linguagem natural, Curso em Curitiba')
teste

"""# Etapa 4: Pré-processamento da base de dados

### Limpeza dos textos
"""

# Exibir os 10 primeiros registros
base_dados.head(10)

# Aplicação da função "preprocessamento"
base_dados['questao'] = base_dados['questao'].apply(preprocessamento)

# Exibir os 10 primeiros registros, agora pré-processado
base_dados.head(10)

"""### Tratamento da classe"""

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
len(base_dados_final)

# Exibindo um exemplo da frase e dicionário
base_dados_final[0]

# Exibindo um exemplo da frase
base_dados_final[0][0]

# Exibindo um exemplo do dicionário
base_dados_final[0][1]

# Exibindo o tipo do dado
type(base_dados_final[0][1])

# Exibir nossa base de dados:
base_dados_final

"""# Etapa 5: Criação do classificador"""

# Agora que nossa base de dados está no formato requerido pelo spaCy,
# vamos criar nosso modelo (algoritmo) que está trabalhando com um rede neural (spaCy)

# Inicializar um novo modelo em branco:
modelo = spacy.blank('pt')
# Definir o tipo de caracterização do modelo, no caso, "textual"
categorias = modelo.create_pipe("textcat")
# Adicionar o rótulo lembrar
categorias.add_label("FACIL")
# Adicionar o rótulo entender
categorias.add_label("MEDIA")
# Adicionar o rótulo aplicar
categorias.add_label("DIFICIL")
# Adicionar...
modelo.add_pipe(categorias)
# Armazenar os resultados
historico = []
# Assim, nosso classificador (baseado nessa rede neural) estará nessa variável "modelo"

# Iniciar nosso treinamento
modelo.begin_training()
# Definir a quantidade de épocas q o algoritmo será executado, no caso, 1000:
for epoca in range(1000):
  # Misturar os dados, pq nossa base de dados possuia "alegria" no início e "medo" no fim
  random.shuffle(base_dados_final)
  # Criar o dicionário (onde controlaremos o erro) 
  losses = {}
  # De quanto em quanto registros faremos a atualização dos pesos:
  # Como possuímos 196 registros, divididos por 30, teremos 6,5 ciclos, ou seja, vai pegando de 30 em 30 registros e submete à rede neural,
  # faz o cálculo do erro, depois faz o ajuste dos pesos. Até completar a 1º época. 
  # Depois repete isso até completar as 1000 épocas
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

# Observe que temos essa lista e dicionário:
historico

# Vamos adicionar somente os valores numéricos em uma lista historico_loss
historico_loss = []
for i in historico:
  historico_loss.append(i.get('textcat'))

# Conversão para o tipo NumPy, para depois passarmos para a visualização do gráfico
historico_loss = np.array(historico_loss)
historico_loss

# Importar 
import matplotlib.pyplot as plt
# Plotar
plt.plot(historico_loss)
# Personalizar 
plt.title('Progressão do erro')
plt.xlabel('Épocas')
plt.ylabel('Erro')
# Observer como o erro está dminuindo! Chega a um erro muito próximo de zero!!
# Nesse gráfico, podemos perceber que não é necessário fazer o treinamento por tantas épocas.
# No caso, na época em torno de 200-300 ele já estabiliza e não temos mais um ganho
# Útil para eu verificar por quantas épocas devo treinar o meu modelo

# Vamos salvar o modelo em disco temporário, ao fechar será perdido!! 
# Observe o diretório "modelo" criado na aba lateral do Colab contendo o "modelo"
# modelo.to_disk("modelo")
modelo.to_disk("/content/drive/My Drive/Cursos - recursos/spaCy")

"""# Etapa 6: Testes com uma frase"""

# Carregar o modelo que salvamos (para levar pronto para outra máquina e não termos que executar as etapas anteriores novamente):
# A partir do diretório "modelo" na aba lateral do Colab
modelo_carregado = spacy.load("/content/drive/My Drive/Cursos - recursos/spaCy")
modelo_carregado

# Criar uma variável para exemplo
texto_positivo = 'Qual é a fórmula da energia?'

# Para passar essa variável para nossa rede neural, é necessário executarmos o pré-processamento dos dados:
texto_positivo = preprocessamento(texto_positivo)
texto_positivo

# Vamos passar nosso texto de exemplo para nosso modelo:
previsao = modelo_carregado(texto_positivo)
previsao

# Para visualizar os resultados da previsão:
previsao.cats

# Criar um exemplo negativo
texto_negativo = 'Qual associação em séries de pilhas fornece diferença de potencial, nas condições-padrão, suficiente para acender o LED azul?'
# Posso já passar o texto pré-processado
previsao = modelo_carregado(preprocessamento(texto_negativo))
previsao.cats

"""# Etapa 7: Avaliação do modelo

## Avaliação na base de treinamento
"""

previsoes = []
# Percorrer todos os textos em nossa base de dados:
for texto in base_dados['questao']:
  #print(texto)
  # Passar cada uma das frases para nosso modelo e registrar em previsão
  previsao = modelo_carregado(texto)
  # Adicionar as categorias das previsões dentro da variável previsões
  previsoes.append(previsao.cats)

# Exibir o registro das previsões 
previsoes

# Para finalizar, vou formatar nossos dados deixando: texto - categoria

previsoes_final = []
# Para cada registro em nossa variável:
for previsao in previsoes:
  # Se o valor de alegria for maior que medo, registrar alegria   
  if(   (previsao['DIFICIL'] >= previsao['MEDIA']  ) 
    and (previsao['DIFICIL'] >= previsao['FACIL'] )):
    previsoes_final.append('dificil')

  elif( (previsao['MEDIA'] >= previsao['DIFICIL']  ) 
    and (previsao['MEDIA'] >= previsao['FACIL'] )):
    previsoes_final.append('media')

  elif( (previsao['FACIL'] >= previsao['DIFICIL']  ) 
    and (previsao['FACIL'] >= previsao['FACIL'] )):
    previsoes_final.append('facil')
           
  else:
    previsoes_final.append('0')

previsoes_final = np.array(previsoes_final)

# Exibir os valores previstos de cada frase:
previsoes_final

# Exibir as categorias reais da base de dados
respostas_reais = base_dados['categoria'].values
respostas_reais

# Comparativo entre o valor real e o previsto
from sklearn.metrics import confusion_matrix, accuracy_score
# (100% de acerto pq são dados de treinamento
accuracy_score(respostas_reais, previsoes_final)

# Gerar a matriz de confusão )
cm = confusion_matrix(respostas_reais, previsoes_final)
cm
