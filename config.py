"""
DetectAI - Parâmetros do projeto
================================
Todos os números "fixos" do projeto ficam aqui, num só lugar, para o grupo poder
explicar e ajustar sem procurar pelo código. Os outros arquivos só leem estes valores.

Para mudar um parâmetro: edite o número, salve e clique em Rerun (ou reabra o programa).
"""
import os

# ----------------------------------------------------------------------
# Pastas e arquivos
# ----------------------------------------------------------------------
PASTA = os.path.dirname(os.path.abspath(__file__))        # pasta do projeto
PASTA_DADOS = os.path.join(PASTA, "dataset")              # exemplos coletados e rotulados
PASTA_MODELO = os.path.join(PASTA, "modelo")              # modelo treinado, relatório e figuras
ARQ_MODELO = os.path.join(PASTA_MODELO, "modelo_grupo.joblib")

# ----------------------------------------------------------------------
# Leitura do vídeo e pré-processamento (Etapa 2)
# ----------------------------------------------------------------------
PASSO = 2              # analisa 1 a cada 2 quadros (quanto maior, mais rápido)
LARGURA_PADRAO = 640   # largura (px) para a qual o quadro é redimensionado
DESFOQUE_PADRAO = 5    # kernel do desfoque gaussiano (ímpar; 1 = sem desfoque)

# ----------------------------------------------------------------------
# Segmentação: subtração de fundo, sombras e limpeza (Etapa 3)
# ----------------------------------------------------------------------
HISTORY = 300          # MOG2: quantos quadros o modelo de fundo "lembra"
PRE_APRENDIZADO = 40   # quadros lidos ANTES de começar, só para o MOG2 aprender o fundo
RIGOR_SOMBRA = 0.92    # pixel com brilho até 92% do fundo pode ser sombra
KERNEL_MORF = 5        # tamanho do "pincel" das operações morfológicas
ITERACOES_MORF = 2     # quantas vezes a limpeza morfológica é repetida
MUDANCA_LUZ = 0.45     # se mais de 45% do quadro "se mexeu", é mudança de luz: o quadro é ignorado
QUADROS_ADAPTACAO = 15 # após uma mudança de luz, o MOG2 aprende mais rápido por 15 quadros
TAXA_ADAPTACAO = 0.05  # taxa de aprendizado do MOG2 durante essa adaptação

# ----------------------------------------------------------------------
# Caixas: unir, filtrar e separar objetos grudados
# ----------------------------------------------------------------------
DIST_MESCLAR = 15      # caixas a menos de 15 px uma da outra viram uma só
MESCLA_MAX = 2.0       # só une duas caixas se a caixa unida não ficar mais que 2x as duas somadas
AREA_MAX = 0.30        # caixas maiores que 30% do quadro são descartadas (não são um objeto)

# Separação de objetos grudados (transformada de distância + watershed)
SEPARAR_NUCLEOS = (0.5, 0.6, 0.7, 0.8)  # limites testados para o "núcleo" de cada objeto
SEPARAR_FRACAO = 0.2     # cada parte separada precisa ter ao menos 20% da mancha
SEPARAR_EMPILHADO = 0.5  # partes uma em cima da outra (>= 50% de sobreposição) são o mesmo objeto
SEPARAR_FECHAMENTO = 7   # fechamento da máscara fina antes de separar (tapa buracos do objeto)

# ----------------------------------------------------------------------
# Reconhecimento Inteligente (YOLO, modelo externo)
# ----------------------------------------------------------------------
MODELO_YOLO = "yolov8s.pt"  # "s" (small) reconhece melhor que o "n" (nano), mas é mais lento
CONF_YOLO = 0.25       # confiança mínima do YOLO (menor = reconhece mais, erra mais)
YOLO_TAMANHO = 640     # tamanho da imagem enviada ao YOLO (maior = enxerga objetos menores)
YOLO_INTERVALO = 2     # o YOLO roda em 1 a cada 2 quadros analisados; no outro, reaproveita o resultado
GATE_MOVIMENTO = 0.002 # o YOLO só roda se pelo menos 0,2% do quadro estiver em movimento
MOVIMENTO_MIN = 0.05   # fração da caixa do objeto que precisa estar em movimento

# Classes do YOLO (conjunto COCO) que interessam: nome em inglês -> (nome exibido, grupo).
# Os nomes exibidos não têm acento porque o OpenCV não desenha acentos.
YOLO_CLASSES = {
    "person": ("Pessoa", "Pessoas"),
    "bicycle": ("Bicicleta", "Veículos"),
    "car": ("Carro", "Veículos"),
    "motorcycle": ("Moto", "Veículos"),
    "bus": ("Onibus", "Veículos"),
    "truck": ("Caminhao", "Veículos"),
    "bird": ("Passaro", "Animais"),
    "cat": ("Gato", "Animais"),
    "dog": ("Cachorro", "Animais"),
    "horse": ("Cavalo", "Animais"),
    "sheep": ("Ovelha", "Animais"),
    "cow": ("Vaca", "Animais"),
    "elephant": ("Elefante", "Animais"),
    "bear": ("Urso", "Animais"),
    "zebra": ("Zebra", "Animais"),
    "giraffe": ("Girafa", "Animais"),
}

# ----------------------------------------------------------------------
# Modelo treinado pelo grupo: coleta de exemplos e treino automático
# ----------------------------------------------------------------------
VOTOS_MINIMOS = 3        # votos de um objeto antes de fixar o nome
VOTOS_ESQUECIMENTO = 0.8   # a cada quadro, os votos antigos do nome valem 80% do que valiam
COLETA_INTERVALO = 5     # salva um recorte do mesmo objeto a cada 5 quadros analisados
COLETA_MAX_OBJETO = 15   # no máximo 15 recortes por objeto (evita exemplos repetidos)
BLOCO_DENTRO = 0.6        # coleta: um objeto do YOLO "está no bloco" se 60% da caixa dele está dentro
BLOCO_COBERTURA = 0.5     # coleta: os objetos precisam cobrir pelo menos 50% do bloco de movimento
CERTEZA_AUTOMATICA = 50  # treino automático: certeza mínima (%) do YOLO para aceitar um rótulo

# Grupo exibido -> pasta de rótulo, e pasta de rótulo -> nome na interface.
PASTA_DO_GRUPO = {"Pessoas": "pessoa", "Veículos": "veiculo", "Animais": "animal"}
NOME_DA_PASTA = {"pessoa": "Pessoa", "veiculo": "Veículo", "animal": "Animal", "outro": "Outro",
                 "descartar": "Descartar"}

# ----------------------------------------------------------------------
# Exibição: estabilização das caixas e imagens na tela
# ----------------------------------------------------------------------
GRUPOS = ("Pessoas", "Veículos", "Animais", "Outros")
CORES = {                         # cor de cada grupo, em BGR (o OpenCV usa Azul-Verde-Vermelho)
    "Pessoas": (0, 220, 0),       # verde
    "Veículos": (255, 140, 0),    # azul
    "Animais": (0, 180, 255),     # laranja
    "Outros": (170, 170, 170),    # cinza
}

# Estabilização das caixas (só na exibição; a detecção não muda)
SUAVE_POSICAO = 0.5    # quanto a caixa segue a nova medida a cada quadro (1 = sem suavizar)
SUAVE_VELOCIDADE = 0.2 # quanto a velocidade estimada se corrige a cada quadro
SUAVE_TAMANHO = 0.3    # quanto a largura e a altura seguem a nova medida
PERSISTENCIA = 3       # quadros que a caixa continua na tela depois que o objeto some (não pisca)
CONFIRMACAO = 2        # quadros seguidos que um objeto novo precisa aparecer antes de ganhar caixa

PROPORCAO_FINAL = [2, 3]  # divisão da última linha entre os contornos (etapa 9) e o resultado (etapa 10)
# Objetos parecidos com sombra (ex.: ônibus ou carro cinza sobre asfalto cinza): a
# carroceria passa no teste de sombra e só sobram o letreiro e as rodas.
# MOG2: quanto do "peso" do fundo uma cor precisa ter para ser considerada fundo.
# O padrão do OpenCV (0,9) faz a traseira de um ônibus longo e de cor uniforme virar
# fundo antes de ele terminar de passar (a caixa pega só a metade da frente).
# Com 0,7 uma cor nova leva cerca de 3 vezes mais quadros para virar fundo.
# Logo depois de uma mudança de luz volta a 0,9, para o fundo novo ser aceito rápido.
FUNDO_RAZAO = 0.7
RECUPERAR_ESCUROS = True
ESCURO_FRACAO = 0.35     # recupera se menos de 35% da região sobreviveu à remoção de sombras
ESCURO_AREA_MIN = 0.004  # só regiões com pelo menos 0,4% do quadro

# Divisão de grupos (modo Treinado pelo grupo): pessoas lado a lado viram um bloco
# só na máscara; o bloco é cortado entre as cabeças, se o modelo concordar.
DIVIDIR_GRUPOS = True
DIVIDIR_CLASSES = ("Pessoa",)    # só grupos de pessoas são divididos (um veículo cortado ao meio
                                 # teria as duas metades reconhecidas como veículo)
DIVIDIR_PROPORCAO = (0.5, 3.5)   # só blocos com largura/altura nessa faixa (uma pessoa sozinha é ~0,4)
DIVIDIR_QUEDA = 0.08     # entre duas cabeças, a silhueta desce pelo menos 8% da altura do bloco
DIVIDIR_LARGURA = 0.15   # cada parte tem pelo menos 15% da altura do bloco de largura
DIVIDIR_MARGEM = 0.05    # as partes podem ter até 5 pontos de certeza a menos que o bloco inteiro

TEMPO_REAL = True        # na tela, pula quadros quando o computador não acompanha o vídeo
PULOS_MAX = 3            # no máximo 3 quadros analisados pulados seguidos (o rastreamento não se perde)
FPS_ETAPAS = 4           # as imagens das etapas 1 a 9 são atualizadas no máximo 4 vezes por segundo
FPS_TELA = 8           # no máximo 8 atualizações da tela por segundo (o processamento não para)
QUALIDADE_JPEG = 80    # qualidade das imagens enviadas à tela (100 = máxima, ~6x mais pesada)
LARG_ETAPA = 360       # largura (px) das imagens das etapas 1 a 9 enviadas à tela
