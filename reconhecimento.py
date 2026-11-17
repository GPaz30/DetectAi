"""
DetectAI - Reconhecimento: dar nome às caixas
=============================================
A detecção diz ONDE há movimento; este arquivo diz O QUE é cada objeto.
Dois reconhecedores, escolhidos na interface:

  ReconhecedorYolo  - "Inteligente (YOLO)": rede neural pronta, treinada por
                      terceiros (modelo EXTERNO). Só aceita objetos em movimento.
  ReconhecedorGrupo - "Treinado pelo grupo": modelo treinado pela equipe
                      (treinar_modelo.py) com as características de cada caixa.
"""
import os

import cv2
import numpy as np

import caracteristicas as car
import config as cfg
import deteccao as det
from rastreamento import Rastreador


# ----------------------------------------------------------------------
# Carregar os modelos
# ----------------------------------------------------------------------
def carregar_yolo():
    """Carrega o YOLO (o arquivo do modelo é baixado sozinho na primeira vez)."""
    from ultralytics import YOLO
    return YOLO(cfg.MODELO_YOLO)


def carregar_modelo_grupo():
    """Lê o modelo treinado pelo grupo; devolve None se ainda não houver um."""
    if not os.path.exists(cfg.ARQ_MODELO):
        return None
    import joblib
    pacote = joblib.load(cfg.ARQ_MODELO)
    usar_um_nucleo(pacote["modelo"])
    return pacote


def usar_um_nucleo(modelo):
    """
    Faz o modelo prever usando um núcleo só do processador.

    O treino usa todos os núcleos (n_jobs=-1), o que acelera a floresta
    aleatória. Mas no vídeo o modelo é consultado a cada quadro, com poucas
    caixas por vez: dividir esse trabalho pequeno entre os núcleos custa mais
    do que ele próprio e deixava o vídeo várias vezes mais lento.
    """
    paralelos = {k: 1 for k in modelo.get_params() if k == "n_jobs" or k.endswith("__n_jobs")}
    if paralelos:
        modelo.set_params(**paralelos)


# ----------------------------------------------------------------------
# Inteligente (YOLO)
# ----------------------------------------------------------------------
class ReconhecedorYolo:
    """
    O YOLO diz O QUE é cada objeto, e a máscara de movimento diz se ele está SE
    MOVENDO. Só ficam os objetos reconhecidos E em movimento; assim, sombras e
    carros parados são descartados.

    Rastreamento + votação: o rastreador do YOLO (ByteTrack) dá um número a cada
    objeto, e o nome é a classe mais votada ao longo dos quadros; assim um veículo
    não fica trocando entre "Carro" e "Onibus".

    'todo_quadro=True' (usado na coleta de exemplos) faz o YOLO rodar em todos os
    quadros analisados; senão, ele roda em 1 a cada YOLO_INTERVALO.
    """

    def __init__(self, modelo, todo_quadro=False):
        self.modelo = modelo
        self.modelo.predictor = None        # descarta o rastreador de uma execução anterior
        self.intervalo = 1 if todo_quadro else cfg.YOLO_INTERVALO
        self.rastreio = True                # False se o ByteTrack não estiver disponível
        self.votos = {}                     # {id do objeto: {classe: soma das confianças}}
        self.quadros_com_movimento = 0
        self.reconhecidos = []              # último resultado (reaproveitado entre as execuções)

    def reconhecer(self, frame, caixas, bruta, passos=1):
        """
        Devolve (reconhecidos, nao_reconhecidos):
          reconhecidos     - objetos que o YOLO reconheceu e que estão em movimento;
          nao_reconhecidos - caixas de movimento que ele não reconheceu ("Movimento").
        'passos' > 1: quadros foram pulados desde o anterior (reprodução em tempo
        real); as caixas da última execução estariam fora do lugar, então o YOLO
        roda de novo neste quadro.
        """
        # Máscara "leve" (sem a abertura morfológica): preserva pessoas pequenas
        # e distantes, que a limpeza forte apagaria.
        leve = cv2.dilate(cv2.medianBlur(bruta, 5), np.ones((3, 3), np.uint8), iterations=2)

        # O YOLO é a parte mais pesada: só roda se há movimento no quadro.
        if cv2.countNonZero(leve) < cfg.GATE_MOVIMENTO * leve.size:
            self.reconhecidos = []
        else:
            if self.quadros_com_movimento % self.intervalo == 0 or passos > 1:
                self.reconhecidos = self._detectar(frame, leve)
            self.quadros_com_movimento += 1

        nao_reconhecidos = [{"caixa": cx, "nome": "Movimento", "grupo": "Outros"} for cx in caixas
                            if all(det.fracao_dentro(cx, o["caixa"]) < 0.3 for o in self.reconhecidos)]
        return self.reconhecidos, nao_reconhecidos

    def _executar_yolo(self, frame):
        """Roda o YOLO com rastreamento; se o rastreador falhar, segue sem ele."""
        opcoes = dict(verbose=False, conf=cfg.CONF_YOLO, imgsz=cfg.YOLO_TAMANHO)
        if self.rastreio:
            try:
                return self.modelo.track(frame, persist=True, tracker="bytetrack.yaml", **opcoes)[0]
            except Exception:
                self.rastreio = False
        return self.modelo(frame, **opcoes)[0]

    def _detectar(self, frame, mascara):
        resultado = self._executar_yolo(frame)

        # Números dos objetos dados pelo rastreador (None sem rastreio ou sem objetos).
        ids = resultado.boxes.id
        lista_ids = ids.int().tolist() if ids is not None else [None] * len(resultado.boxes)

        objetos = []
        for b, tid in zip(resultado.boxes, lista_ids):
            nome_en = resultado.names[int(b.cls[0])]
            if nome_en not in cfg.YOLO_CLASSES:       # classe que não interessa (ex.: banco, semáforo)
                continue
            conf = float(b.conf[0])

            # Votação: soma a confiança de cada classe vista para este objeto.
            if tid is not None:
                votos = self.votos.setdefault(tid, {})
                votos[nome_en] = votos.get(nome_en, 0.0) + conf
                nome_en = max(votos, key=votos.get)   # classe mais votada até agora

            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])   # caixa dentro do quadro
            w, h = x2 - x1, y2 - y1
            if w < 8 or h < 8:
                continue

            # Quanto da caixa do YOLO tem pixels em movimento na máscara?
            regiao = mascara[y1:y2, x1:x2]
            if regiao.size and (regiao > 0).mean() >= cfg.MOVIMENTO_MIN:
                nome, grupo = cfg.YOLO_CLASSES[nome_en]
                objetos.append({"caixa": (x1, y1, w, h), "nome": nome, "grupo": grupo, "conf": conf})
        return objetos


# ----------------------------------------------------------------------
# Treinado pelo grupo
# ----------------------------------------------------------------------
class ReconhecedorGrupo:
    """
    Calcula as características de cada caixa de movimento (e a forma, HOG) e
    pergunta ao modelo treinado pelo grupo qual é a classe. O nome final é
    decidido por votação ao longo dos quadros, pesada pela certeza do modelo.

    Sem modelo treinado ('pacote' None), todas as caixas saem como "Outro".
    """

    def __init__(self, pacote):
        self.pacote = pacote
        self.rastreador = Rastreador()

    def reconhecer(self, frame, caixas, limpa, passos=1, fina=None):
        """
        Devolve (objetos, ids, características); os dois últimos servem à coleta de exemplos.
        'passos': quadros analisados desde a chamada anterior (ver Rastreador.atualizar).
        'fina': máscara sem a dilatação, usada para achar o vão entre as cabeças de
        um grupo de pessoas (a limpeza morfológica tapa esse vão).
        """
        ids = self.rastreador.atualizar(caixas, frame.shape[1], passos)
        feats = [car.extrair(cx, limpa, frame, self.rastreador.velocidade(t)) for cx, t in zip(caixas, ids)]
        if not caixas:
            return [], ids, feats

        previstos, probabilidades = self._prever(frame, caixas, feats)
        objetos = []
        for cx, tid, nome, prob in zip(caixas, ids, previstos, probabilidades):
            partes = self._dividir(frame, limpa, fina if fina is not None else limpa, cx, tid, str(nome), prob)
            if partes is None:
                nome = self.rastreador.votar(tid, str(nome), prob)
                objetos.append({"caixa": cx, "nome": nome, "grupo": car.GRUPOS.get(nome, "Outros")})
                continue
            # Bloco dividido: o voto do objeto vai para a classe das partes.
            classe, prob_partes, caixas_partes = partes
            nome = self.rastreador.votar(tid, classe, prob_partes)
            if nome == classe:
                objetos += [{"caixa": p, "nome": nome, "grupo": car.GRUPOS.get(nome, "Outros")} for p in caixas_partes]
            else:   # a votação ainda não mudou de ideia: mostra o bloco inteiro
                objetos.append({"caixa": cx, "nome": nome, "grupo": car.GRUPOS.get(nome, "Outros")})
        return objetos, ids, feats

    def _dividir(self, frame, limpa, fina, caixa, tid, nome, prob):
        """
        Testa se o bloco é um grupo de objetos lado a lado (ex.: 3 pessoas juntas,
        que pelo formato pareciam um veículo).

        A geometria propõe os cortes (deteccao.dividir_por_silhueta) e o MODELO DO
        GRUPO decide: o corte só vale se todas as partes forem reconhecidas como a
        MESMA classe, e essa classe for uma das DIVIDIR_CLASSES (só Pessoa), com
        certeza parecida com a do bloco inteiro. Veículos nunca são cortados: as
        duas metades de um carro seriam reconhecidas como veículo.

        Devolve (classe, chances médias das partes, caixas das partes) ou None.
        """
        if not cfg.DIVIDIR_GRUPOS or self.pacote is None:
            return None
        caixas_partes = det.dividir_por_silhueta(fina, caixa)
        if caixas_partes is None:
            return None

        velocidade = self.rastreador.velocidade(tid)          # o grupo anda junto
        feats = [car.extrair(p, limpa, frame, velocidade) for p in caixas_partes]
        previstos, probs = self._prever(frame, caixas_partes, feats)
        classe = str(previstos[0])
        if classe not in cfg.DIVIDIR_CLASSES or any(str(p) != classe for p in previstos):
            return None

        if probs[0] is None:                                  # modelo sem chances: só a classe conta
            return classe, None, caixas_partes
        media = {c: float(np.mean([p[c] for p in probs])) for c in probs[0]}
        # Classe diferente da do bloco: as partes precisam convencer tanto quanto ele.
        if classe != nome and media[classe] < max(prob.values()) - cfg.DIVIDIR_MARGEM:
            return None
        return classe, media, caixas_partes

    def _prever(self, frame, caixas, feats):
        """Classe prevista para cada caixa e, quando o modelo informa, a chance de cada classe."""
        if self.pacote is None:
            return ["Outro"] * len(caixas), [None] * len(caixas)

        X = np.array([[f[n] for n in self.pacote["atributos"]] for f in feats], dtype=np.float32)
        if self.pacote.get("usa_hog"):
            # Forma do objeto (HOG) da região dentro da caixa, igual ao treino.
            hog = np.array([car.aparencia(car.recorte_caixa(frame, cx)) for cx in caixas])
            X = np.hstack([X, hog])

        modelo = self.pacote["modelo"]
        if not hasattr(modelo, "predict_proba"):
            return modelo.predict(X), [None] * len(caixas)
        # Uma consulta só ao modelo: a classe prevista é a de maior chance.
        chances = modelo.predict_proba(X)                # <- o modelo do grupo decide
        classes = [str(c) for c in modelo.classes_]
        previstos = [classes[i] for i in chances.argmax(axis=1)]
        return previstos, [dict(zip(classes, linha)) for linha in chances]
