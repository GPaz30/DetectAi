"""
DetectAI - Processamento de um vídeo, quadro a quadro
=====================================================
Junta as peças dos outros arquivos na ordem em que acontecem:

    deteccao.py (onde há movimento) -> reconhecimento.py (o que é cada objeto)
    -> exemplos.py (coleta, se pedida) -> rastreamento.py (caixas estáveis)

Não tem interface: a mesma função serve para mostrar o vídeo na tela (app.py)
e para coletar exemplos sem exibir nada (treino automático).
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config as cfg
import deteccao as det
from rastreamento import EstabilizadorCaixas
from reconhecimento import ReconhecedorGrupo, ReconhecedorYolo


@dataclass
class Opcoes:
    """Escolhas feitas na interface para processar um vídeo."""
    largura: int = cfg.LARGURA_PADRAO      # largura do quadro redimensionado (px)
    desfoque: int = cfg.DESFOQUE_PADRAO    # kernel do desfoque gaussiano
    sensibilidade: int = 5                 # 1 a 10: maior = detecta movimentos mais sutis
    area_min: int = 400                    # tamanho mínimo de uma região em movimento (px²)
    usa_mog: bool = True                   # True = MOG2; False = diferença de quadros
    grupos: tuple = cfg.GRUPOS             # grupos que aparecem no resultado


@dataclass
class Resultado:
    """Tudo o que foi calculado para um quadro: as imagens de cada etapa e os objetos."""
    n: int                    # número do quadro no vídeo (começa em 1)
    total: int                # total de quadros do vídeo
    original: np.ndarray      # etapa 1
    redim: np.ndarray         # etapa 2
    cinza: np.ndarray         # etapa 3
    suave: np.ndarray         # etapa 4
    fundo: np.ndarray         # etapa 5: saída da subtração de fundo
    sombras: np.ndarray       # etapa 6: sombras removidas (None na diferença de quadros)
    sem_sombras: np.ndarray   # etapa 7: máscara de movimento sem sombras
    limpa: np.ndarray         # etapa 8: máscara depois da limpeza morfológica
    cortes: np.ndarray        # etapa 9: linhas onde objetos grudados foram separados
    caixas: list              # etapa 9: caixas de movimento
    objetos: list             # etapa 10: [{"caixa", "nome", "grupo"}], já estabilizados
    mudanca_de_luz: bool      # True se o quadro foi ignorado por mudança de luz
    passos: int = 1           # quadros analisados desde o anterior (> 1 se a reprodução em tempo real pulou)

    def contagem(self):
        """Quantos objetos de cada grupo há neste quadro."""
        return {g: sum(1 for o in self.objetos if o["grupo"] == g) for g in cfg.GRUPOS}


class Processador:
    """
    Processa os quadros de UM vídeo, guardando o que precisa lembrar entre eles
    (modelo de fundo, rastreadores, votos).

    'yolo'    - modelo YOLO carregado: reconhecimento Inteligente. Sem ele, usa o
                modelo treinado pelo grupo ('pacote'; None = ainda não treinado).
    'coletor' - se informado, salva exemplos de treino durante o processamento.
    """

    def __init__(self, opcoes, yolo=None, pacote=None, coletor=None):
        self.opcoes = opcoes
        self.coletor = coletor
        self.segmentador = det.Segmentador(opcoes.usa_mog, opcoes.sensibilidade)
        # Na coleta, o YOLO roda em todos os quadros: a caixa de cada exemplo
        # precisa ser a do próprio quadro.
        self.yolo = ReconhecedorYolo(yolo, todo_quadro=coletor is not None) if yolo is not None else None
        self.grupo = ReconhecedorGrupo(pacote) if yolo is None else None
        self.estabilizador = EstabilizadorCaixas()

    def aprender_fundo(self, quadro):
        """Pré-aprendizado: mostra um quadro ao modelo de fundo, sem detectar nada."""
        self.segmentador.aprender(det.pre_processar(quadro, self.opcoes.largura, self.opcoes.desfoque)[2])

    def processar(self, quadro, n, total, passos=1):
        """
        Passa um quadro por todas as etapas e devolve o Resultado.
        'passos' = quadros analisados desde o anterior: 1, ou mais quando a
        reprodução em tempo real pulou quadros (ver processar_video).
        """
        o = self.opcoes

        # Etapas 1 a 4: pré-processamento.
        redim, cinza, suave, suave_cinza = det.pre_processar(quadro, o.largura, o.desfoque)

        # Etapas 5 a 8: máscara de movimento, sem sombras e limpa.
        fundo, sombras, bruta = self.segmentador.aplicar(suave, suave_cinza)
        limpa = det.refinar_mascara(bruta)

        # Mudança de luz: o quadro inteiro "se mexeu"; não há caixas neste quadro.
        luz = det.mudanca_de_luz(limpa)
        if luz:
            self.segmentador.avisar_mudanca_de_luz()

        # Etapa 9: caixas de movimento.
        cortes = np.zeros_like(limpa)
        caixas = []
        fina = det.mascara_fina(limpa, bruta)     # máscara sem a dilatação: contorno mais fiel ao objeto
        if not luz:
            caixas = det.detectar_caixas(limpa, o.area_min, fina, cortes, suave)

        # Etapa 10: nome de cada objeto, filtro por grupo e caixas estáveis.
        objetos = self._reconhecer(n, redim, caixas, bruta, limpa, not luz, passos, fina)
        objetos = [x for x in objetos if x["grupo"] in o.grupos]
        objetos = self.estabilizador.atualizar(objetos, passos)

        return Resultado(n, total, quadro, redim, cinza, suave, fundo, sombras, bruta, limpa,
                         cortes, caixas, objetos, luz, passos)

    def _reconhecer(self, n, frame, caixas, bruta, limpa, coletar, passos=1, fina=None):
        """Dá nome aos objetos e, se houver coletor, salva os exemplos do quadro."""
        coletor = self.coletor if coletar else None

        if self.yolo is not None:
            reconhecidos, nao_reconhecidos = self.yolo.reconhecer(frame, caixas, bruta, passos)
            if coletor is not None:
                coletor.coletar_yolo(n, frame, limpa, reconhecidos, nao_reconhecidos, caixas)
            return reconhecidos + nao_reconhecidos

        objetos, ids, feats = self.grupo.reconhecer(frame, caixas, limpa, passos, fina)
        if coletor is not None:
            coletor.coletar_movimento(n, frame, caixas, ids, feats, self.grupo.rastreador)
        return objetos


def informacoes_do_video(caminho):
    """Largura, altura, FPS e número de quadros do vídeo; None se ele não puder ser aberto."""
    cap = cv2.VideoCapture(caminho)
    try:
        if not cap.isOpened():
            return None
        return {"largura": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "altura": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
                "quadros": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    finally:
        cap.release()


def ler_quadro(caminho, indice):
    """Lê um quadro do vídeo pelo índice (começa em 0); None se não conseguir."""
    cap = cv2.VideoCapture(caminho)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, quadro = cap.read()
        return quadro if ok else None
    finally:
        cap.release()


def processar_video(caminho, opcoes, yolo=None, pacote=None, coletor=None, tempo_real=False):
    """
    Gerador: lê o vídeo e entrega um Resultado para cada quadro analisado
    (1 a cada PASSO quadros). Uso:

        for resultado in processar_video("video.mp4", Opcoes()):
            ...

    tempo_real=True (usado na tela): o vídeo anda na velocidade em que foi gravado.
      - Se o computador ficar para trás do relógio do vídeo, quadros são pulados
        (no máximo PULOS_MAX seguidos, para o rastreamento não se perder), em vez
        de o vídeo passar em câmera lenta;
      - se ficar adiantado, espera, em vez de o vídeo passar acelerado.
    A coleta de exemplos usa tempo_real=False: todos os quadros são analisados.
    """
    cap = cv2.VideoCapture(caminho)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        processador = Processador(opcoes, yolo, pacote, coletor)

        # Pré-aprendizado: o MOG2 vê os primeiros quadros só para conhecer o fundo;
        # assim a detecção já vale desde o primeiro quadro.
        if opcoes.usa_mog:
            for _ in range(cfg.PRE_APRENDIZADO):
                ok, quadro = cap.read()
                if not ok:
                    break
                processador.aprender_fundo(quadro)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # volta ao início do vídeo

        n = 0
        inicio = None        # instante (relógio do computador) em que o vídeo "começou"
        pulados = 0          # quadros analisados pulados desde o último processado
        while True:
            # grab() só avança o vídeo; o quadro só é convertido em imagem (retrieve)
            # se for mesmo analisado.
            if not cap.grab():          # acabou o vídeo
                break
            n += 1
            if (n - 1) % cfg.PASSO != 0:    # pula quadros para ganhar velocidade
                continue

            if tempo_real:
                agora = time.monotonic()
                if inicio is None:
                    inicio = agora - (n - 1) / fps
                atraso = (agora - inicio) - (n - 1) / fps      # > 0: o computador está atrasado
                if atraso > cfg.PASSO / fps and pulados < cfg.PULOS_MAX:
                    pulados += 1
                    continue
                if atraso < 0:
                    time.sleep(-atraso)

            ok, quadro = cap.retrieve()
            if not ok:
                break
            yield processador.processar(quadro, n, total, pulados + 1)
            pulados = 0
    finally:
        cap.release()
