"""
DetectAI - Detecção de movimento (processamento de imagem)
==========================================================
Tudo o que transforma um quadro do vídeo em CAIXAS de movimento (etapas 2 a 9):

    quadro -> redimensionar -> desfoque gaussiano -> subtração de fundo (MOG2 ou
    diferença de quadros) -> remoção de sombras (HSV) -> limpeza morfológica ->
    contornos -> filtro de área -> unir caixas -> separar objetos grudados

Aqui não há interface nem reconhecimento: só OpenCV e NumPy. Dar NOME às caixas
é trabalho do reconhecimento.py.
"""
import cv2
import numpy as np

import config as cfg


# ----------------------------------------------------------------------
# Etapa 2 - Pré-processamento
# ----------------------------------------------------------------------
def redimensionar(frame, largura):
    """Muda a largura do quadro mantendo a proporção (a altura é calculada)."""
    h, w = frame.shape[:2]
    if w == largura:
        return frame
    nova_altura = int(h * largura / w)
    # INTER_AREA é a interpolação indicada quando se está reduzindo a imagem.
    return cv2.resize(frame, (largura, nova_altura), interpolation=cv2.INTER_AREA)


def pre_processar(frame, largura, desfoque):
    """
    Prepara o quadro para a detecção. Devolve 4 versões:
      redim        - quadro redimensionado (colorido)
      cinza        - redimensionado em tons de cinza
      suave        - redimensionado + desfoque gaussiano (colorido)
      suave_cinza  - desfoque gaussiano em tons de cinza
    """
    redim = redimensionar(frame, largura)

    # O kernel do desfoque gaussiano precisa ser ímpar (3, 5, 7...).
    k = desfoque if desfoque % 2 == 1 else desfoque + 1
    # O desfoque suaviza o ruído da câmera. k = 1 significa "sem desfoque".
    suave = cv2.GaussianBlur(redim, (k, k), 0) if k > 1 else redim

    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)
    suave_cinza = cv2.cvtColor(suave, cv2.COLOR_BGR2GRAY)
    return redim, cinza, suave, suave_cinza


# ----------------------------------------------------------------------
# Etapa 3 - Segmentação: o que está se mexendo?
# ----------------------------------------------------------------------
def diferenca_quadros(atual, anterior, limiar):
    """
    Método próprio em NumPy (alternativa ao MOG2): marca como movimento todo
    pixel cuja diferença para o quadro anterior é maior que o limiar.
    """
    if anterior is None:  # primeiro quadro: ainda não há com o que comparar
        return np.zeros_like(atual)
    # Converte para int16 antes de subtrair; com uint8 (0-255) a subtração
    # "daria a volta" (ex.: 10 - 20 = 246) e estragaria o resultado.
    diff = np.abs(atual.astype(np.int16) - anterior.astype(np.int16))
    return ((diff > limiar) * 255).astype(np.uint8)  # máscara: 255 = movimento


def remover_sombras(frame_bgr, fundo_bgr, mascara, v_max, s_tol=60, h_tol=25):
    """
    Remove sombras da máscara usando o espaço de cores HSV.
    Uma sombra escurece o fundo (V diminui) mas mantém quase a mesma cor (H) e a
    mesma saturação (S); um objeto real costuma mudar a cor. Os pixels do
    primeiro plano que se comportam como sombra são apagados.

    Devolve a máscara sem sombras e uma imagem com o que foi removido.
    """
    f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    b = cv2.cvtColor(fundo_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    razao_v = f[..., 2] / (b[..., 2] + 1e-6)  # brilho em relação ao fundo (1e-6 evita divisão por zero)
    dif_s = np.abs(f[..., 1] - b[..., 1])     # diferença de saturação
    dif_h = np.abs(f[..., 0] - b[..., 0])     # diferença de matiz
    dif_h = np.minimum(dif_h, 180 - dif_h)    # o matiz é circular (0 a 180): pega o menor caminho

    # Sombra: mais escuro que o fundo (35% a v_max) e com a cor parecida.
    sombra = (razao_v >= 0.35) & (razao_v <= v_max) & (dif_s <= s_tol) & (dif_h <= h_tol)

    removida = sombra & (mascara > 0)  # só importa a sombra que estava na máscara
    saida = mascara.copy()
    saida[removida] = 0
    return saida, (removida * 255).astype(np.uint8)


def recuperar_objetos_escuros(objeto, sombras):
    """
    Devolve ao objeto a parte que a remoção de sombras apagou por engano.

    Um ônibus ou carro cinza, um pouco mais escuro que o asfalto, passa no teste
    de sombra (mais escuro, mesma cor): a carroceria e as janelas são apagadas e
    sobram só o letreiro e as rodas, em pedaços. Para separar isso de uma sombra
    de verdade, usa-se onde cada coisa fica:

      - sombra no chão fica na BASE do objeto (embaixo ou ao lado), nunca acima;
      - o que sobrou do objeto (letreiro, rodas) marca a largura dele e até onde
        ele vai para baixo.

    Para cada região em movimento (objeto + sombra juntos), se menos de
    ESCURO_FRACAO dela sobreviveu como objeto, os pixels de "sombra" que estão
    entre as laterais do que sobrou e ACIMA do ponto mais baixo dele voltam a
    ser objeto. A sombra embaixo ou ao lado continua removida. Em uma pessoa ou
    um carro colorido, em que a maior parte sobrevive, nada muda.

    Devolve (objeto, sombras) atualizados.
    """
    tudo = cv2.bitwise_or(objeto, sombras)
    base = cv2.dilate(cv2.medianBlur(tudo, 5), np.ones((3, 3), np.uint8), iterations=2)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(base)
    # O que sobrou do objeto, sem pontinhos e sem linhas finas: a borda de uma
    # sombra (penumbra) não passa no teste de sombra e sobra como uma linha fina;
    # se ela contasse, a sombra inteira no chão seria devolvida ao objeto.
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sobrou = cv2.morphologyEx(cv2.medianBlur(objeto, 5), cv2.MORPH_OPEN, nucleo) > 0
    area_min = cfg.ESCURO_AREA_MIN * objeto.size
    objeto, sombras = objeto.copy(), sombras.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < area_min:
            continue
        regiao = rotulos[y:y + h, x:x + w] == i
        restos = sobrou[y:y + h, x:x + w] & regiao
        if not restos.any():                               # só sombra: é sombra mesmo
            continue
        colunas, linhas = np.where(restos.any(axis=0))[0], np.where(restos.any(axis=1))[0]
        c0, c1, base_objeto = colunas[0], colunas[-1] + 1, linhas[-1] + 1
        janela = (tudo[y:y + base_objeto, x + c0:x + c1] > 0) & regiao[:base_objeto, c0:c1]
        n_janela = int(janela.sum())
        if n_janela == 0 or restos[:base_objeto, c0:c1].sum() >= cfg.ESCURO_FRACAO * n_janela:
            continue                                       # a maior parte sobreviveu: nada a fazer
        objeto[y:y + base_objeto, x + c0:x + c1][janela] = 255
        sombras[y:y + base_objeto, x + c0:x + c1][janela] = 0
    return objeto, sombras


class Segmentador:
    """
    Gera, para cada quadro, a máscara de movimento (branco = movimento).

    Dois métodos, escolhidos na interface:
      - MOG2 (OpenCV): compara o quadro com um modelo do fundo aprendido ao longo
        do vídeo; marca as sombras, que depois são removidas (também em HSV);
      - diferença de quadros (código do grupo): compara com o quadro anterior.

    A sensibilidade (1 a 10) vira o limiar de cada método: maior = mais sensível.
    """

    def __init__(self, usa_mog, sensibilidade):
        self.mog = None
        if usa_mog:
            # detectShadows=True faz o MOG2 marcar as sombras com o valor 127.
            self.mog = cv2.createBackgroundSubtractorMOG2(
                history=cfg.HISTORY, varThreshold=int(46 - 4 * sensibilidade), detectShadows=True)
            self.mog.setBackgroundRatio(cfg.FUNDO_RAZAO)     # objetos longos não viram fundo (ver config.py)
        self.limiar_diferenca = int(50 - 4 * sensibilidade)
        self.anterior = None       # quadro anterior (diferença de quadros)
        self.adaptando = 0         # quadros restantes de aprendizado rápido (após mudança de luz)

    def aprender(self, suave):
        """Pré-aprendizado: mostra um quadro ao MOG2 só para ele conhecer o fundo."""
        if self.mog is not None:
            self.mog.apply(suave)

    def aplicar(self, suave, suave_cinza):
        """
        Devolve (fundo, sombras, bruta):
          fundo   - saída da subtração de fundo (MOG2: 255 = movimento, 127 = sombra)
          sombras - o que foi apagado por ser sombra (None na diferença de quadros)
          bruta   - máscara de movimento sem sombras, antes da limpeza
        """
        if self.mog is None:
            bruta = diferenca_quadros(suave_cinza, self.anterior, self.limiar_diferenca)
            self.anterior = suave_cinza
            return bruta, None, bruta

        # Depois de uma mudança de luz, o MOG2 aprende mais rápido por alguns
        # quadros, para absorver logo o fundo novo. (-1 = taxa automática)
        # Nesses quadros o fundo novo também precisa ser ACEITO rápido: usa a razão
        # padrão do OpenCV (0,9); no resto do tempo, FUNDO_RAZAO (ver config.py).
        taxa = cfg.TAXA_ADAPTACAO if self.adaptando > 0 else -1
        self.mog.setBackgroundRatio(0.9 if self.adaptando > 0 else cfg.FUNDO_RAZAO)
        self.adaptando = max(self.adaptando - 1, 0)
        fundo = self.mog.apply(suave, learningRate=taxa)

        # Fica só o primeiro plano "certo" (255): o 127 do MOG2 é sombra.
        _, bruta = cv2.threshold(fundo, 200, 255, cv2.THRESH_BINARY)
        # Remoção de sombras em HSV: apaga as sombras que o MOG2 não marcou.
        bruta, sombras_hsv = remover_sombras(suave, self.mog.getBackgroundImage(), bruta, cfg.RIGOR_SOMBRA)
        sombras = np.where((fundo == 127) | (sombras_hsv > 0), 255, 0).astype(np.uint8)
        if cfg.RECUPERAR_ESCUROS:
            bruta, sombras = recuperar_objetos_escuros(bruta, sombras)
        return fundo, sombras, bruta

    def avisar_mudanca_de_luz(self):
        """Chamado quando o quadro foi ignorado por mudança de luz: acelera o aprendizado do fundo."""
        self.adaptando = cfg.QUADROS_ADAPTACAO


def refinar_mascara(mascara):
    """
    Limpeza morfológica (etapa 8):
      1) mediana: tira pontinhos isolados ("sal e pimenta");
      2) abertura (erosão + dilatação): remove manchas pequenas;
      3) fechamento (dilatação + erosão): tapa buracos dentro do objeto;
      4) dilatação: engrossa um pouco, para o objeto não ficar "magro".
    """
    m = cv2.medianBlur(mascara, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.KERNEL_MORF, cfg.KERNEL_MORF))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=cfg.ITERACOES_MORF)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=cfg.ITERACOES_MORF + 1)
    return cv2.dilate(m, kernel, iterations=1)


def mascara_fina(limpa, bruta):
    """
    Pixels que se mexeram de verdade dentro das manchas limpas, sem a borda que a
    dilatação acrescenta. Usada para deixar as caixas justas e separar objetos.
    """
    return cv2.morphologyEx(cv2.bitwise_and(limpa, bruta), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def mudanca_de_luz(limpa):
    """
    True se boa parte do quadro "se mexeu" de uma vez (uma nuvem, uma lâmpada
    acendendo): não é movimento de objetos, e o quadro deve ser ignorado.
    """
    return cv2.countNonZero(limpa) > cfg.MUDANCA_LUZ * limpa.size


# ----------------------------------------------------------------------
# Geometria das caixas. Uma caixa é (x, y, largura, altura).
# ----------------------------------------------------------------------
def iou(a, b):
    """Sobreposição entre duas caixas: área em comum / área total das duas (0 a 1)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / (aw * ah + bw * bh - iw * ih)


def fracao_dentro(a, b):
    """Qual fração da área da caixa 'a' está dentro da caixa 'b' (0 a 1)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / float(aw * ah)


def mesclar_caixas(caixas, area_max_px=None):
    """
    Une caixas que se tocam ou estão a menos de DIST_MESCLAR pixels uma da outra,
    para uma pessoa não aparecer dividida em várias caixas.

    Para não criar caixas gigantes, duas caixas só são unidas se a caixa unida:
      - não for mais que MESCLA_MAX vezes a soma das duas áreas (senão são objetos
        diferentes, e a caixa unida seria quase toda fundo);
      - não passar de 'area_max_px'.
    """
    dist = cfg.DIST_MESCLAR
    caixas = [list(c) for c in caixas]
    mudou = True
    while mudou:                      # repete até nenhuma união nova acontecer
        mudou = False
        saida = []
        while caixas:
            x, y, w, h = caixas.pop()
            i = 0
            while i < len(caixas):
                x2, y2, w2, h2 = caixas[i]
                # As caixas, "engordadas" em 'dist', se sobrepõem?
                perto = (x < x2 + w2 + dist and x2 < x + w + dist and
                         y < y2 + h2 + dist and y2 < y + h + dist)
                # Como ficaria a caixa unida (a menor que contém as duas)?
                nx, ny = min(x, x2), min(y, y2)
                nw, nh = max(x + w, x2 + w2) - nx, max(y + h, y2 + h2) - ny
                cabe = (nw * nh <= cfg.MESCLA_MAX * (w * h + w2 * h2) and
                        (area_max_px is None or nw * nh <= area_max_px))
                if perto and cabe:
                    x, y, w, h = nx, ny, nw, nh
                    caixas.pop(i)
                    mudou = True
                else:
                    i += 1
            saida.append([x, y, w, h])
        caixas = saida
    return [tuple(c) for c in caixas]


def ajustar_caixa(caixa, fina, parte=None):
    """
    Encolhe a caixa até os pixels que REALMENTE se mexeram ('fina'). A limpeza
    morfológica engorda as manchas; sem este ajuste, a caixa fica com uma borda
    de fundo em volta do objeto.
    Com 'parte' (máscara de um objeto separado), usa só os pixels dessa parte,
    para não pegar o objeto vizinho. Devolve None se não houver pixels suficientes.
    """
    x, y, w, h = caixa
    regiao = fina[y:y + h, x:x + w]
    if parte is not None:
        regiao = (regiao > 0).astype(np.uint8) & parte.astype(np.uint8)
    pts = cv2.findNonZero(regiao)
    if pts is None or len(pts) < 20:
        return None
    fx, fy, fw, fh = cv2.boundingRect(pts)
    if fw < 8 or fh < 8:
        return None
    return (x + fx, y + fy, fw, fh)


# ----------------------------------------------------------------------
# Separação de objetos grudados (transformada de distância + watershed)
# ----------------------------------------------------------------------
def _unir_partes_empilhadas(partes):
    """
    Partes uma em cima da outra (cabeça e corpo, tronco e pernas) são o mesmo
    objeto: junta as que se sobrepõem na horizontal. A largura de cada parte
    ignora 5% de pixels soltos de cada lado.
    """
    def faixa_x(m):
        xs = np.where(m)[1]
        return np.percentile(xs, 5), np.percentile(xs, 95)

    mudou = True
    while mudou and len(partes) > 1:
        mudou = False
        for a in range(len(partes)):
            for b in range(a + 1, len(partes)):
                (a0, a1), (b0, b1) = faixa_x(partes[a]), faixa_x(partes[b])
                sobreposicao = min(a1, b1) - max(a0, b0)
                if sobreposicao >= cfg.SEPARAR_EMPILHADO * min(a1 - a0, b1 - b0):
                    partes[a] = partes[a] | partes.pop(b)
                    mudou = True
                    break
            if mudou:
                break
    return partes


def _dividir(reg, pad, dist, img_pad, limite, area, area_min):
    """
    Passos 2 a 4 de separar_objetos para um limite de núcleo.
    Devolve a lista de máscaras das partes, ou None se a mancha não for dividida.
    """
    # 2) núcleos: cada pedaço separado é a semente de um objeto
    nucleo = (dist >= limite * dist.max()).astype(np.uint8)
    n, sementes = cv2.connectedComponents(nucleo)
    if n - 1 < 2:                                  # um núcleo só: um objeto só
        return None

    # 3) watershed (como no exemplo clássico do OpenCV): sementes = 2..n, fundo
    # certo (longe da mancha) = 1, região a decidir = 0. A inundação acontece sobre
    # a IMAGEM do quadro, então a divisão segue as bordas de cor entre os objetos.
    marcas = np.zeros(pad.shape, np.int32)
    fundo_certo = cv2.dilate(pad, np.ones((3, 3), np.uint8), iterations=3) == 0
    marcas[fundo_certo] = 1
    marcas[nucleo == 1] = sementes[nucleo == 1] + 1
    if img_pad is not None:
        marcas = cv2.watershed(img_pad, marcas)[2:-2, 2:-2]
    else:
        # sem imagem: cada pixel vai para a semente mais próxima
        distancias = np.stack([cv2.distanceTransform((sementes != k).astype(np.uint8), cv2.DIST_L2, 3)
                               for k in range(1, n)])
        marcas = (np.argmin(distancias, axis=0) + 2).astype(np.int32)[2:-2, 2:-2]

    # 4) ficam as partes de bom tamanho; cada pixel da mancha que sobrou (linhas
    # do watershed, partes pequenas) vai para a parte grande mais próxima.
    minimo = max(area_min, cfg.SEPARAR_FRACAO * area)
    grandes = [m for m in ((marcas == k) & (reg == 1) for k in range(2, n + 1)) if m.sum() >= minimo]
    if len(grandes) < 2:
        return None
    distancias = np.stack([cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) for m in grandes])
    mais_perto = np.argmin(distancias, axis=0)
    partes = [(reg == 1) & (mais_perto == i) for i in range(len(grandes))]

    partes = _unir_partes_empilhadas(partes)
    return partes if len(partes) >= 2 else None


def separar_objetos(mascara, caixa, area_min, cortes=None, imagem=None):
    """
    Divide uma mancha em objetos separados quando ela é, na verdade, dois ou mais
    objetos grudados (ex.: duas pessoas andando lado a lado).

    Técnica clássica de processamento de imagem (a do exemplo de watershed do OpenCV):
      1) TRANSFORMADA DE DISTÂNCIA: cada pixel da mancha recebe a distância até a
         borda mais próxima. O "miolo" de cada objeto fica com valores altos; a
         ligação estreita entre dois objetos (o "pescoço") fica com valores baixos;
      2) NÚCLEOS: os pixels com distância >= um limite x a maior distância. Cada
         pedaço de núcleo separado é a semente de um objeto. Testa os limites de
         SEPARAR_NUCLEOS, do menor para o maior: quanto mais largo o pescoço,
         maior o limite necessário para as sementes se separarem;
      3) WATERSHED ("divisor de águas"): a partir das sementes, a imagem é
         "inundada", e as linhas onde as águas de sementes diferentes se encontram
         dividem os objetos;
      4) FILTROS: partes pequenas e partes uma em cima da outra (cabeça e corpo)
         são juntadas de volta: só separa objetos lado a lado e de bom tamanho.

    'mascara' deve ser pouco engordada (ver detectar_caixas): na máscara da etapa 8,
    a limpeza morfológica cola objetos vizinhos e apaga o "pescoço".
    'imagem' é o quadro colorido (sem ela, cada pixel vai para a semente mais próxima).
    'cortes' (opcional) recebe as linhas de divisão, para a etapa 9.

    Devolve [(caixa, máscara_da_parte)]; a máscara é None quando a mancha não foi dividida.
    """
    x, y, w, h = caixa
    inteira = [(caixa, None)]
    reg = (mascara[y:y + h, x:x + w] > 0).astype(np.uint8)
    area = int(reg.sum())
    if area < 2 * area_min:
        return inteira

    # 1) transformada de distância (com 2 px de margem em volta da caixa)
    pad = cv2.copyMakeBorder(reg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(pad, cv2.DIST_L2, 5)
    if dist.max() < 3:
        return inteira
    img_pad = None
    if imagem is not None:
        img_pad = cv2.copyMakeBorder(imagem[y:y + h, x:x + w], 2, 2, 2, 2, cv2.BORDER_REPLICATE)

    partes = None
    for limite in cfg.SEPARAR_NUCLEOS:
        partes = _dividir(reg, pad, dist, img_pad, limite, area, area_min)
        if partes is not None:
            break
    if partes is None:
        return inteira

    if cortes is not None:
        # A linha de divisão fica onde duas partes se encostam.
        vizinhas = sum(cv2.dilate(p.astype(np.uint8), np.ones((3, 3), np.uint8)) for p in partes)
        cortes[y:y + h, x:x + w][(vizinhas >= 2) & (reg == 1)] = 255

    saida = []
    for m in partes:
        px, py, pw, ph = cv2.boundingRect(m.astype(np.uint8))
        if pw >= 8 and ph >= 8:
            saida.append(((x + px, y + py, pw, ph), m))
    return saida if len(saida) >= 2 else inteira


# ----------------------------------------------------------------------
# Etapa 9 - Da máscara às caixas
# ----------------------------------------------------------------------
def detectar_caixas(limpa, area_min, fina=None, cortes=None, imagem=None):
    """
    Transforma a máscara limpa em caixas, uma por objeto:
      1) contornos das manchas; os menores que 'area_min' são descartados;
      2) caixas próximas são unidas (pedaços do mesmo objeto);
      3) caixas gigantes são descartadas (não são um objeto);
      4) objetos grudados são separados (watershed);
      5) cada caixa é ajustada aos pixels que realmente se mexeram ('fina').

    'cortes' (opcional) recebe as linhas onde as manchas foram divididas;
    'imagem' (opcional) é o quadro colorido, usado pelo watershed.
    """
    area_max_px = cfg.AREA_MAX * limpa.shape[0] * limpa.shape[1]

    contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caixas = []
    for c in contornos:
        if cv2.contourArea(c) >= area_min:      # filtro de área mínima
            x, y, w, h = cv2.boundingRect(c)    # menor retângulo que envolve o contorno
            if w >= 12 and h >= 12:             # ignora faixas muito finas
                caixas.append((x, y, w, h))

    caixas = mesclar_caixas(caixas, area_max_px)
    caixas = [c for c in caixas if c[2] * c[3] <= area_max_px]
    if fina is None:
        return [cx for c in caixas for cx, _ in separar_objetos(limpa, c, area_min, cortes, imagem)]

    # A separação usa a máscara fina com um fechamento leve: os buracos de cada
    # objeto são tapados, mas o espaço entre dois objetos vizinhos continua aberto.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.SEPARAR_FECHAMENTO, cfg.SEPARAR_FECHAMENTO))
    base = cv2.morphologyEx(fina, cv2.MORPH_CLOSE, kernel)
    final = []
    for c in caixas:
        for cx, parte in separar_objetos(base, c, area_min, cortes, imagem):
            if parte is None:
                final.append(ajustar_caixa(cx, fina) or cx)
            else:
                final.append(ajustar_caixa(c, fina, parte) or cx)
    return final


# ----------------------------------------------------------------------
# Divisão de grupos pela silhueta
# ----------------------------------------------------------------------
def dividir_por_silhueta(mascara, caixa):
    """
    Pessoas andando lado a lado viram UMA mancha na máscara, e o bloco largo
    acaba parecendo um veículo. Esta função procura onde cortá-lo.

    Para cada coluna do bloco mede-se onde começa o movimento, de cima para
    baixo: é a SILHUETA de cima. Cada cabeça forma um pico, e entre duas
    cabeças a silhueta desce até os ombros (um "vale"). O bloco é cortado no
    ponto mais baixo de cada vale que tenha pelo menos DIVIDIR_QUEDA da altura.

    Devolve a lista de caixas das partes (2 ou mais), ou None se não há corte.
    Quem decide se o corte vale é o modelo do grupo (ReconhecedorGrupo).
    """
    x, y, w, h = caixa
    if h < 20 or not (cfg.DIVIDIR_PROPORCAO[0] <= w / h <= cfg.DIVIDIR_PROPORCAO[1]):
        return None
    regiao = mascara[y:y + h, x:x + w] > 0
    if not regiao.any():
        return None

    # Topo de cada coluna (linha do primeiro pixel de movimento; h se a coluna está vazia).
    tem = regiao.any(axis=0)
    topo = np.where(tem, regiao.argmax(axis=0), h).astype(np.float32)
    k = max(3, (w // 20) | 1)                                  # suaviza o contorno (ímpar)
    topo = cv2.medianBlur(topo.reshape(1, -1), min(k, 5)).ravel() if w >= 5 else topo
    topo = np.convolve(np.pad(topo, k // 2, mode="edge"), np.ones(k) / k, mode="valid")

    largura_min = max(int(cfg.DIVIDIR_LARGURA * h), 4)
    queda_min = cfg.DIVIDIR_QUEDA * h

    # Picos = pontos mais altos da silhueta (menor linha), separados por pelo
    # menos largura_min; os mais altos têm prioridade.
    candidatos = [c for c in range(w) if (c == 0 or topo[c] <= topo[c - 1]) and (c == w - 1 or topo[c] < topo[c + 1])]
    picos = []
    for c in sorted(candidatos, key=lambda c: topo[c]):
        if all(abs(c - p) >= largura_min for p in picos):
            picos.append(c)
    picos.sort()

    # Um corte no ponto mais baixo entre dois picos vizinhos, se o vale for fundo.
    cortes, inicio = [], 0
    for p1, p2 in zip(picos, picos[1:]):
        v = p1 + int(np.argmax(topo[p1:p2 + 1]))
        queda = topo[v] - max(topo[p1], topo[p2])
        if queda >= queda_min and v - inicio >= largura_min and w - v >= largura_min:
            cortes.append(v)
            inicio = v
    if not cortes:
        return None

    # Caixa justa de cada parte.
    partes = []
    for a, b in zip([0] + cortes, cortes + [w]):
        colunas = regiao[:, a:b]
        linhas, cols = np.where(colunas.any(axis=1))[0], np.where(colunas.any(axis=0))[0]
        if len(linhas) and len(cols):
            partes.append((x + a + int(cols[0]), y + int(linhas[0]),
                           int(cols[-1] - cols[0] + 1), int(linhas[-1] - linhas[0] + 1)))
    return partes if len(partes) >= 2 else None
