"""
DetectAI - Características das regiões em movimento
====================================================
O modelo do grupo não recebe a imagem: recebe NÚMEROS que descrevem cada região
em movimento. Este arquivo calcula esses números, e é usado em três lugares:
  - exemplos.py: na coleta dos exemplos de treino;
  - treinar_modelo.py: para treinar e avaliar o modelo;
  - reconhecimento.py: para classificar com o modelo treinado.

Ficar num arquivo só garante que a coleta, o treino e o uso calculam as
características EXATAMENTE do mesmo jeito (se fossem diferentes, o modelo
receberia no uso números que nunca viu no treino).

Conteúdo:
  NOMES_ATRIBUTOS  - nomes das 10 características, na ordem usada pelo modelo
  extrair()        - calcula as 10 características de uma caixa
  aparencia()      - descritor HOG: a "forma" do objeto dentro da caixa
  regra_manual()   - a antiga classificação "Simples" por regras, mantida só
                     como REFERÊNCIA para comparar com o modelo treinado
"""
import cv2
import numpy as np

# ----------------------------------------------------------------------
# Classes que o modelo aprende
# ----------------------------------------------------------------------
# Nome da pasta de rotulação -> nome exibido no vídeo (sem acento: limitação do OpenCV)
CLASSES = {
    "pessoa": "Pessoa",
    "veiculo": "Veiculo",
    "animal": "Animal",
    "outro": "Outro",      # sombras, reflexos, folhas, ruído... qualquer coisa que não seja as outras
}
# Nome exibido -> grupo da interface (cor da caixa e filtro "Mostrar")
GRUPOS = {"Pessoa": "Pessoas", "Veiculo": "Veículos", "Animal": "Animais", "Outro": "Outros"}

# Características, na ordem em que o modelo as recebe.
NOMES_ATRIBUTOS = [
    "razao",          # largura / altura da caixa (pessoa < 1, carro > 1)
    "preenchimento",  # fração da caixa ocupada pela máscara (carro é "cheio")
    "solidez",        # área da mancha / área do seu contorno convexo (pernas e falhas diminuem)
    "larg_norm",      # largura / posição do pé da caixa (tamanho corrigido pela perspectiva)
    "alt_norm",       # altura / posição do pé da caixa
    "pos_y",          # posição do pé da caixa na imagem (0 = topo, 1 = base)
    "area_rel",       # área da caixa / área do quadro
    "velocidade",     # deslocamento do centro por quadro, em fração da largura do quadro
    "saturacao",      # saturação média (HSV) dentro da mancha: cores vivas x cinza
    "contraste",      # desvio padrão do cinza dentro da mancha: textura/detalhe
]


# ----------------------------------------------------------------------
# Extração das características
# ----------------------------------------------------------------------
def extrair(caixa, mascara, frame_bgr, velocidade):
    """
    Calcula as características de UMA caixa.
      caixa      - (x, y, largura, altura)
      mascara    - máscara de movimento limpa (255 = movimento)
      frame_bgr  - quadro colorido redimensionado
      velocidade - velocidade do objeto, medida pelo Rastreador (rastreamento.py)
    Devolve um dicionário {nome_da_característica: valor}.
    """
    H, W = mascara.shape[:2]
    x, y, w, h = caixa
    m = mascara[y:y + h, x:x + w]
    pix = m > 0
    area_mancha = float(pix.sum())

    # Solidez: compara a maior mancha com o seu contorno convexo
    # (um "elástico" em volta dela). Formas com reentrâncias têm solidez menor.
    solidez = 0.0
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        c = max(contornos, key=cv2.contourArea)
        area_hull = cv2.contourArea(cv2.convexHull(c))
        solidez = cv2.contourArea(c) / area_hull if area_hull > 0 else 0.0

    # Cor e textura medidas só nos pixels em movimento (não no fundo da caixa).
    recorte = frame_bgr[y:y + h, x:x + w]
    saturacao = contraste = 0.0
    if area_mancha > 0:
        hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        saturacao = float(hsv[..., 1][pix].mean()) / 255.0
        cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
        contraste = float(cinza[pix].std()) / 255.0

    base = max(y + h, 0.25 * H)   # posição do pé (evita dividir por valor muito pequeno)
    return {
        "razao": w / float(h),
        "preenchimento": area_mancha / float(w * h),
        "solidez": solidez,
        "larg_norm": w / base,
        "alt_norm": h / base,
        "pos_y": (y + h) / float(H),
        "area_rel": (w * h) / float(W * H),
        "velocidade": float(velocidade),
        "saturacao": saturacao,
        "contraste": contraste,
    }


# ----------------------------------------------------------------------
# Aparência: descritor HOG (Histograma de Gradientes Orientados)
# ----------------------------------------------------------------------
# O HOG divide a região em pequenas células e, em cada uma, conta em quais
# direções estão as bordas. O resultado descreve a FORMA do objeto (cabeça e
# pernas de uma pessoa, rodas e teto de um carro), não a cor. É um descritor
# clássico de visão computacional (Dalal e Triggs, 2005, para detectar pessoas).
#
# Configuração: a região é reduzida para 32x32 px, cinza; células de 8x8 px,
# blocos de 2x2 células sem sobreposição, 9 direções -> 2x2 blocos x 4 células x 9 = 144 números.
# (Testamos também 324 e 36 números: 144 acertou igual ou mais e treina mais rápido.)
#
# O HOG é calculado aqui mesmo, com NumPy, em vez de usar cv2.HOGDescriptor:
# algumas versões do OpenCV não trazem mais essa função, e assim o projeto
# funciona com qualquer versão.
TAM_HOG = 32
CELULA = 8
BLOCO = 2            # células por lado em cada bloco
DIRECOES = 9         # faixas de direção entre 0 e 180 graus
N_HOG = (TAM_HOG // (CELULA * BLOCO)) ** 2 * BLOCO * BLOCO * DIRECOES
NOMES_HOG = [f"hog_{i}" for i in range(N_HOG)]


def _hog(cinza):
    """
    HOG de uma imagem cinza 32x32:
      1) gradiente de cada pixel (quanto o brilho muda na horizontal e na vertical);
      2) cada pixel vota na faixa da direção do seu gradiente, com peso = força da borda;
         o voto é dividido entre as duas faixas vizinhas (interpolação);
      3) um histograma de 9 faixas por célula de 8x8 px;
      4) as células são agrupadas em blocos de 2x2 e normalizadas, para o
         resultado não depender de a imagem estar mais clara ou mais escura.
    """
    img = cinza.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)       # filtro [-1, 0, 1]
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    forca = np.sqrt(gx ** 2 + gy ** 2)
    angulo = np.rad2deg(np.arctan2(gy, gx)) % 180.0       # direção sem sentido: 0 a 180 graus

    largura_faixa = 180.0 / DIRECOES
    pos = angulo / largura_faixa - 0.5                    # posição contínua entre as faixas
    f0 = np.floor(pos).astype(int)
    peso1 = pos - f0                                       # parte do voto que vai para a faixa seguinte
    f0 %= DIRECOES
    f1 = (f0 + 1) % DIRECOES

    n_cel = TAM_HOG // CELULA
    cel_y, cel_x = np.indices(img.shape) // CELULA
    hist = np.zeros((n_cel, n_cel, DIRECOES), np.float32)
    np.add.at(hist, (cel_y, cel_x, f0), forca * (1 - peso1))
    np.add.at(hist, (cel_y, cel_x, f1), forca * peso1)

    blocos = []
    for by in range(0, n_cel, BLOCO):
        for bx in range(0, n_cel, BLOCO):
            v = hist[by:by + BLOCO, bx:bx + BLOCO].ravel()
            v = v / np.sqrt((v ** 2).sum() + 1e-6)          # normalização L2
            v = np.minimum(v, 0.2)                           # limita bordas muito fortes (L2-Hys)
            v = v / np.sqrt((v ** 2).sum() + 1e-6)
            blocos.append(v)
    return np.concatenate(blocos).astype(np.float32)


# Margem em volta da caixa nos recortes salvos pela coleta (fração do maior lado).
MARGEM_RECORTE = 0.3


def recorte_caixa(imagem, caixa):
    """
    Parte da imagem dentro da caixa, sem a linha da borda (1 px de cada lado).
    Os recortes da coleta têm um retângulo verde desenhado exatamente na borda;
    tirando 1 px, treino e app analisam exatamente a mesma região.
    """
    x, y, w, h = caixa
    if w <= 4 or h <= 4:
        return imagem[y:y + h, x:x + w]
    return imagem[y + 1:y + h - 1, x + 1:x + w - 1]


def caixa_no_recorte(x, y, w, h):
    """Posição da caixa dentro do recorte salvo pela coleta (que tem uma margem)."""
    m = int(MARGEM_RECORTE * max(w, h))
    return (x - max(x - m, 0), y - max(y - m, 0), w, h)


def aparencia(regiao_bgr, espelhar=False):
    """Descritor HOG (144 números) da região do objeto. 'espelhar' vira a imagem na horizontal."""
    cinza = cv2.cvtColor(regiao_bgr, cv2.COLOR_BGR2GRAY) if regiao_bgr.ndim == 3 else regiao_bgr
    if cinza.size == 0:
        return np.zeros(N_HOG, np.float32)
    cinza = cv2.resize(cinza, (TAM_HOG, TAM_HOG), interpolation=cv2.INTER_AREA)
    if espelhar:
        cinza = cv2.flip(cinza, 1)
    return _hog(cinza)


# ----------------------------------------------------------------------
# Regra manual (o antigo modo "Simples"), usada só como referência
# ----------------------------------------------------------------------
LIM_VEICULO = 0.40
LIM_ANIMAL = 0.22
LIM_PREENCH = 0.62
LIM_VELOC = 0.010


def regra_manual(f):
    """
    Classificação por pontos, com limiares escolhidos à mão (o antigo modo Simples).
    Não é usada no app: serve para o treinamento comparar
    "regras escritas à mão" x "modelo treinado com os dados do grupo".
    """
    r, p, ln, v = f["razao"], f["preenchimento"], f["larg_norm"], f["velocidade"]
    rapido = v >= LIM_VELOC
    meio = (LIM_VEICULO + LIM_ANIMAL) / 2
    pessoa = (4 if (r <= 0.85 and f["alt_norm"] >= 1.2 * ln) else 1 if r <= 1.0 else 0) + (1 if ln <= 0.18 else 0)
    veiculo = (2 if r >= 1.8 else 1 if r >= 1.3 else 0) + (1 if p >= LIM_PREENCH else 0) + \
              (3 if ln >= LIM_VEICULO else 1 if ln >= meio else 0) + (1 if rapido else 0)
    animal = (1 if 0.8 <= r <= 2.6 else 0) + (1 if p < LIM_PREENCH else 0) + \
             (3 if ln <= LIM_ANIMAL else 0) + (1 if (v > 0 and not rapido) else 0)
    pontos = {"Pessoa": pessoa, "Veiculo": veiculo, "Animal": animal}
    nome = max(pontos, key=pontos.get)
    if pontos[nome] < 2:
        return "Outro"
    if veiculo == animal == pontos[nome]:
        return "Veiculo" if ln >= meio else "Animal"
    return nome
