"""
DetectAI - Exemplos de treino (pasta dataset)
=============================================
Cuida dos exemplos que o modelo do grupo usa para aprender:

  coleta     - Coletor: salva um recorte e as características de cada objeto
               enquanto o vídeo é processado;
  rotulação  - o rótulo de um recorte é a pasta em que ele está. Pode ser dado à
               mão, confirmando a sugestão do YOLO, ou de forma automática;
  limpeza    - apagar exemplos ruins.

Estrutura da pasta dataset:
    exemplos.csv         uma linha por recorte: características e sugestão do YOLO
    origem_rotulos.csv   de onde veio cada rótulo (manual, YOLO confirmado, automático...)
    nao_rotulado/        recortes ainda sem rótulo
    pessoa/ veiculo/ animal/ outro/   recortes rotulados
    descartar/           recortes ignorados (ruins ou duvidosos)
"""
import csv
import datetime
import os
import re
import shutil
import time

import cv2
import pandas as pd

import caracteristicas as car
import config as cfg
from deteccao import fracao_dentro
from rastreamento import Rastreador

PASTAS = list(car.CLASSES) + ["nao_rotulado", "descartar"]
COLUNAS = (["arquivo", "video", "quadro", "objeto", "x", "y", "w", "h"] + car.NOMES_ATRIBUTOS
           + ["sugestao", "certeza", "coleta"])


def _recortes(pasta):
    """Nomes dos recortes (.jpg) de uma pasta, em ordem; lista vazia se a pasta não existir."""
    if not os.path.isdir(pasta):
        return []
    return sorted(a for a in os.listdir(pasta) if a.lower().endswith(".jpg"))


def _ler_csv(pasta_dados):
    """Lê o exemplos.csv; devolve None se ele ainda não existir."""
    caminho = os.path.join(pasta_dados, "exemplos.csv")
    return pd.read_csv(caminho) if os.path.exists(caminho) else None


# ----------------------------------------------------------------------
# Coleta
# ----------------------------------------------------------------------
class Coletor:
    """
    Salva exemplos enquanto o vídeo é processado. Para cada objeto:
      - um recorte colorido (com uma margem e um retângulo verde marcando o
        objeto), em dataset/nao_rotulado;
      - uma linha no exemplos.csv com as características e, no modo Inteligente,
        a SUGESTÃO do YOLO e a certeza dela.

    Cada objeto gera no máximo COLETA_MAX_OBJETO recortes: o primeiro no 3º quadro
    em que aparece (quando a velocidade já foi medida) e depois 1 a cada
    COLETA_INTERVALO quadros.

    Uso:  with Coletor("video.mp4") as coletor: ...
    """

    def __init__(self, nome_video, pasta_dados=cfg.PASTA_DADOS):
        self.pasta = os.path.join(pasta_dados, "nao_rotulado")
        os.makedirs(self.pasta, exist_ok=True)
        self.video = nome_video
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(nome_video)[0])[:40]
        self.prefixo = f"{base}_{time.strftime('%Y%m%d%H%M%S')}"   # evita nomes repetidos

        caminho_csv = os.path.join(pasta_dados, "exemplos.csv")
        novo = not os.path.exists(caminho_csv)
        if not novo:
            self._atualizar_csv_antigo(caminho_csv)
        self.arquivo = open(caminho_csv, "a", newline="", encoding="utf-8")
        self.escritor = csv.DictWriter(self.arquivo, fieldnames=COLUNAS)
        if novo:
            self.escritor.writeheader()

        self.rastreador = Rastreador()   # usado quando as caixas vêm do YOLO
        self.rastreador_blocos = Rastreador()   # blocos de movimento com vários objetos (ver coletar_yolo)
        self.salvos = {}                 # recortes já salvos de cada objeto
        self.total = 0

    @staticmethod
    def _atualizar_csv_antigo(caminho_csv):
        """Um exemplos.csv de uma versão anterior pode não ter as colunas mais novas: acrescenta."""
        antigo = pd.read_csv(caminho_csv)
        if "coleta" in antigo.columns:
            return
        if "sugestao" not in antigo.columns:
            antigo["sugestao"], antigo["certeza"] = "", 0.0
        # Recortes com sugestão vieram do YOLO; dos outros, não se sabe.
        antigo["coleta"] = antigo["sugestao"].fillna("").map(lambda s: "yolo" if s else "")
        antigo[COLUNAS].to_csv(caminho_csv, index=False)

    def __enter__(self):
        return self

    def __exit__(self, *erro):
        self.fechar()

    def fechar(self):
        self.arquivo.close()

    def coletar_movimento(self, n_quadro, frame, caixas, ids, feats, rastreador):
        """Modo Treinado pelo grupo: os exemplos são as caixas da detecção de movimento."""
        self._salvar(n_quadro, frame, caixas, ids, feats, rastreador, [("", 0.0)] * len(caixas), "movimento")

    def coletar_yolo(self, n_quadro, frame, limpa, reconhecidos, nao_reconhecidos, blocos=()):
        """
        Modo Inteligente: os exemplos são as caixas da etapa 10. Cada objeto
        reconhecido pelo YOLO leva a classe dele como sugestão; o movimento que
        ele não reconheceu também é salvo, como candidato a "Outro".

        'blocos' são as caixas de movimento da etapa 9. No USO, o modelo do grupo
        não recebe as caixas do YOLO, e sim esses blocos, que podem ser maiores
        que o objeto ou juntar vários (duas pessoas lado a lado viram um bloco
        só, que pelo formato parecia um veículo). Para o modelo aprender com o
        mesmo tipo de caixa que vai receber, cada bloco cujo conteúdo o YOLO
        reconheceu (1 ou mais objetos, todos da MESMA classe) também é salvo,
        como exemplo dessa classe (ver _blocos_de_uma_classe).
        """
        caixas = [o["caixa"] for o in reconhecidos] + [o["caixa"] for o in nao_reconhecidos]
        sugestoes = ([(cfg.PASTA_DO_GRUPO.get(o["grupo"], ""), o.get("conf", 0.0)) for o in reconhecidos]
                     + [("", 0.0)] * len(nao_reconhecidos))
        ids = self.rastreador.atualizar(caixas, frame.shape[1])
        feats = [car.extrair(cx, limpa, frame, self.rastreador.velocidade(t)) for cx, t in zip(caixas, ids)]
        self._salvar(n_quadro, frame, caixas, ids, feats, self.rastreador, sugestoes, "yolo")

        caixas, sugestoes = _blocos_de_uma_classe(blocos, reconhecidos)
        ids = self.rastreador_blocos.atualizar(caixas, frame.shape[1])
        feats = [car.extrair(cx, limpa, frame, self.rastreador_blocos.velocidade(t)) for cx, t in zip(caixas, ids)]
        # Os números dos blocos começam em 100000, para não se misturar com os dos objetos.
        self._salvar(n_quadro, frame, caixas, ids, feats, self.rastreador_blocos, sugestoes, "yolo_bloco",
                     deslocamento=100000)

    def _salvar(self, n_quadro, frame, caixas, ids, feats, rastreador, sugestoes, coleta, deslocamento=0):
        H, W = frame.shape[:2]
        for (x, y, w, h), id_rastreador, f, (sugestao, certeza) in zip(caixas, ids, feats, sugestoes):
            idade = rastreador.idade(id_rastreador)
            if idade < 3 or (idade - 3) % cfg.COLETA_INTERVALO != 0:
                continue
            tid = id_rastreador + deslocamento
            if self.salvos.get(tid, 0) >= cfg.COLETA_MAX_OBJETO:
                continue

            m = int(car.MARGEM_RECORTE * max(w, h))      # margem para ver o contexto
            x0, y0, x1, y1 = max(x - m, 0), max(y - m, 0), min(x + w + m, W), min(y + h + m, H)
            recorte = frame[y0:y1, x0:x1].copy()
            cv2.rectangle(recorte, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (0, 255, 0), 1)
            nome = f"{self.prefixo}_q{n_quadro:05d}_o{tid}.jpg"
            cv2.imwrite(os.path.join(self.pasta, nome), recorte)

            linha = {"arquivo": nome, "video": self.video, "quadro": n_quadro, "objeto": tid,
                     "x": x, "y": y, "w": w, "h": h}
            linha.update({k: round(v, 5) for k, v in f.items()})
            linha.update(sugestao=sugestao, certeza=round(certeza, 3), coleta=coleta)
            self.escritor.writerow(linha)
            self.salvos[tid] = self.salvos.get(tid, 0) + 1
            self.total += 1


def apagar_coleta(prefixo, pasta_dados=cfg.PASTA_DADOS):
    """
    Apaga os exemplos de UMA coleta (Coletor.prefixo) que ainda não foram rotulados.
    Usado quando a coleta de um vídeo é interrompida no meio: sem isso o vídeo
    ficaria com exemplos pela metade e seria pulado na próxima vez.
    """
    pasta = os.path.join(pasta_dados, "nao_rotulado")
    for a in _recortes(pasta):
        if a.startswith(prefixo + "_q"):
            os.remove(os.path.join(pasta, a))
    dados = _ler_csv(pasta_dados)
    if dados is not None:
        da_coleta = dados["arquivo"].astype(str).str.startswith(prefixo + "_q")
        dados[~da_coleta].to_csv(os.path.join(pasta_dados, "exemplos.csv"), index=False)


def _blocos_de_uma_classe(blocos, reconhecidos):
    """
    Blocos de movimento (etapa 9) cujo conteúdo o YOLO reconheceu: 1 ou mais
    objetos, todos da MESMA classe (ex.: duas pessoas lado a lado = "pessoa").
    Devolve as caixas e as sugestões (pasta da classe, menor certeza entre os
    objetos do bloco).

    Ficam de fora: blocos sem objeto do YOLO (já salvos como candidatos a
    Outro), blocos com classes misturadas (pessoa + carro: não há um rótulo
    certo) e blocos em que os objetos cobrem menos de BLOCO_COBERTURA da área
    (o resto seria sombra ou ruído).

    No teste com vídeos sintéticos, estes exemplos fizeram o modelo do grupo
    acertar grupos de pessoas andando juntas em 92% a 100% das vezes (antes,
    34% a 82%), sem piorar os objetos sozinhos.
    """
    caixas, sugestoes = [], []
    for bloco in blocos:
        dentro = [o for o in reconhecidos if fracao_dentro(o["caixa"], bloco) >= cfg.BLOCO_DENTRO]
        if not dentro or len({o["grupo"] for o in dentro}) != 1:
            continue
        cobertura = sum(o["caixa"][2] * o["caixa"][3] for o in dentro) / float(bloco[2] * bloco[3])
        if cobertura < cfg.BLOCO_COBERTURA:
            continue
        caixas.append(bloco)
        sugestoes.append((cfg.PASTA_DO_GRUPO.get(dentro[0]["grupo"], ""), min(o.get("conf", 0.0) for o in dentro)))
    return caixas, sugestoes


# ----------------------------------------------------------------------
# Consultas
# ----------------------------------------------------------------------
def pendentes(pasta_dados=cfg.PASTA_DADOS):
    """Recortes que ainda não foram rotulados."""
    return _recortes(os.path.join(pasta_dados, "nao_rotulado"))


def contar_rotulos(pasta_dados=cfg.PASTA_DADOS):
    """Quantos recortes há em cada pasta (para mostrar o progresso da rotulação)."""
    return {p: len(_recortes(os.path.join(pasta_dados, p))) for p in PASTAS}


def videos_coletados(pasta_dados=cfg.PASTA_DADOS):
    """Nomes dos vídeos que têm exemplos coletados."""
    dados = _ler_csv(pasta_dados)
    return [] if dados is None else sorted(dados["video"].unique())


def objeto_de(arquivo):
    """
    Identifica o objeto de um recorte pelo nome do arquivo:
    <vídeo>_<execução>_q<quadro>_o<objeto>.jpg -> "<vídeo>_<execução>_o<objeto>".
    Recortes com o mesmo identificador são do MESMO objeto, em quadros diferentes.
    """
    if "_q" in arquivo and "_o" in arquivo:
        return arquivo.rsplit("_q", 1)[0] + "_o" + arquivo.rsplit("_o", 1)[1]
    return arquivo


def agrupar_por_objeto(arquivos):
    """Junta os recortes de cada objeto: {identificador do objeto: [recortes]}."""
    grupos = {}
    for a in arquivos:
        grupos.setdefault(objeto_de(a), []).append(a)
    return grupos


# ----------------------------------------------------------------------
# Sugestões do YOLO
# ----------------------------------------------------------------------
def carregar_sugestoes(pasta_dados=cfg.PASTA_DADOS):
    """
    Lê do exemplos.csv o que foi gravado na coleta. Devolve dois dicionários:
      sugestoes   - {recorte: (pasta sugerida pelo YOLO, certeza)}
      viu_o_yolo  - conjunto dos recortes coletados no modo Inteligente
    """
    dados = _ler_csv(pasta_dados)
    if dados is None or "sugestao" not in dados.columns:
        return {}, set()
    com = dados.dropna(subset=["sugestao"])
    sugestoes = {a: (s, float(c)) for a, s, c in zip(com["arquivo"], com["sugestao"], com["certeza"]) if s}
    viu_o_yolo = set(sugestoes)
    if "coleta" in dados.columns:
        viu_o_yolo |= set(dados.loc[dados["coleta"].isin(["yolo", "yolo_bloco"]), "arquivo"])
    return sugestoes, viu_o_yolo


def sugestao_do_objeto(arquivos, sugestoes):
    """
    Sugestão para um objeto inteiro: a classe que o YOLO mais apontou nos recortes
    dele, pesada pela certeza. A certeza do objeto é a média entre TODOS os
    recortes (um recorte sem sugestão conta 0); assim, objetos em que o YOLO
    hesitou ou mudou de ideia ficam com certeza menor.
    Devolve (pasta, certeza), ou ("", 0.0) se o YOLO não reconheceu o objeto.
    """
    votos = {}
    for a in arquivos:
        if a in sugestoes:
            pasta, certeza = sugestoes[a]
            votos[pasta] = votos.get(pasta, 0.0) + certeza
    if not votos:
        return "", 0.0
    melhor = max(votos, key=votos.get)
    return melhor, votos[melhor] / len(arquivos)


# ----------------------------------------------------------------------
# Rotulação
# ----------------------------------------------------------------------
def registrar_origem(arquivos, rotulo, origem, pasta_dados=cfg.PASTA_DADOS):
    """
    Anota de onde veio cada rótulo, em dataset/origem_rotulos.csv:
      manual          - o grupo escolheu a classe;
      yolo_confirmado - o YOLO sugeriu e o grupo confirmou olhando o objeto;
      yolo_lote       - sugestão do YOLO aceita em lote, pela certeza;
      yolo_automatico - treino automático: rótulo dado pelo YOLO, sem revisão.
    Serve para o relatório: quanto da rotulação teve ajuda do YOLO.
    """
    caminho = os.path.join(pasta_dados, "origem_rotulos.csv")
    novo = not os.path.exists(caminho)
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(caminho, "a", encoding="utf-8") as f:
        if novo:
            f.write("arquivo,rotulo,origem,data\n")
        for a in arquivos:
            f.write(f"{a},{rotulo},{origem},{agora}\n")


def contar_origem(arquivos, pasta_dados=cfg.PASTA_DADOS):
    """Quantos dos recortes vieram de cada origem (vale o registro mais recente de cada um)."""
    caminho = os.path.join(pasta_dados, "origem_rotulos.csv")
    ultima = {}
    if os.path.exists(caminho):
        registro = pd.read_csv(caminho)
        ultima = dict(zip(registro["arquivo"], registro["origem"]))
    contagem = {}
    for a in arquivos:
        origem = ultima.get(a, "sem_registro")   # rotulado pelo explorador de arquivos, por exemplo
        contagem[origem] = contagem.get(origem, 0) + 1
    return contagem


def rotular(movimentos, origem, pasta_dados=cfg.PASTA_DADOS):
    """
    Dá o rótulo: move os recortes de nao_rotulado para a pasta da classe.
    'movimentos' é uma lista de (recortes, pasta de destino).
    """
    for arquivos, destino in movimentos:
        os.makedirs(os.path.join(pasta_dados, destino), exist_ok=True)
        for a in arquivos:
            shutil.move(os.path.join(pasta_dados, "nao_rotulado", a), os.path.join(pasta_dados, destino, a))
        registrar_origem(arquivos, destino, origem, pasta_dados)


def desfazer(movimentos, pasta_dados=cfg.PASTA_DADOS):
    """Desfaz uma rotulação: devolve os recortes para nao_rotulado."""
    for arquivos, destino in movimentos:
        for a in arquivos:
            origem = os.path.join(pasta_dados, destino, a)
            if os.path.exists(origem):
                shutil.move(origem, os.path.join(pasta_dados, "nao_rotulado", a))


def sugestoes_em_lote(certeza_min, pasta_dados=cfg.PASTA_DADOS):
    """Objetos pendentes cuja sugestão do YOLO tem pelo menos 'certeza_min' (0 a 1): [(recortes, pasta)]."""
    sugestoes, _ = carregar_sugestoes(pasta_dados)
    lote = []
    for arquivos in agrupar_por_objeto(pendentes(pasta_dados)).values():
        pasta, certeza = sugestao_do_objeto(arquivos, sugestoes)
        if pasta and certeza >= certeza_min:
            lote.append((arquivos, pasta))
    return lote


def rotular_automatico(certeza_min, sem_sugestao_vira_outro=True, pasta_dados=cfg.PASTA_DADOS):
    """
    Rotula sozinho todos os recortes pendentes, usando o YOLO como "professor".
    Um recorte só recebe a classe sugerida se passar em DUAS conferências:
      1) certeza: a sugestão do YOLO para ele tem pelo menos 'certeza_min' (0 a 1);
      2) coerência: é a mesma classe que o YOLO mais apontou para aquele objeto
         ao longo do vídeo (ele não "mudou de ideia").
    Quem não passa vai para "descartar": um exemplo a menos, mas nenhum errado.

    Movimento que o YOLO não reconheceu em NENHUM quadro vira "outro" (sombra,
    reflexo, ruído) se 'sem_sugestao_vira_outro'; senão, é descartado.
    Recortes coletados sem o YOLO (modo Treinado pelo grupo) ficam como estão.

    Devolve {pasta: número de recortes}, com a chave "sem_yolo" para os que ficaram.
    """
    sugestoes, viu_o_yolo = carregar_sugestoes(pasta_dados)
    destinos = {}                                    # {pasta: [recortes]}
    for arquivos in agrupar_por_objeto(pendentes(pasta_dados)).values():
        mais_apontada, _ = sugestao_do_objeto(arquivos, sugestoes)
        for a in arquivos:
            if a in sugestoes:
                pasta, certeza = sugestoes[a]
                destino = pasta if certeza >= certeza_min and pasta == mais_apontada else "descartar"
            elif a not in viu_o_yolo:
                destino = "sem_yolo"
            elif mais_apontada:                      # o YOLO reconheceu o objeto, mas não neste quadro
                destino = "descartar"
            else:
                destino = "outro" if sem_sugestao_vira_outro else "descartar"
            destinos.setdefault(destino, []).append(a)
    rotular([(arquivos, pasta) for pasta, arquivos in destinos.items() if pasta != "sem_yolo"],
            "yolo_automatico", pasta_dados)
    return {pasta: len(arquivos) for pasta, arquivos in destinos.items()}


# ----------------------------------------------------------------------
# Limpeza
# ----------------------------------------------------------------------
def limpar_exemplos(modo, pasta_dados=cfg.PASTA_DADOS):
    """
    Apaga exemplos salvos e mantém o exemplos.csv coerente com os recortes que sobraram.
      modo="grandes"        - exemplos (rotulados ou não) cuja caixa passa de AREA_MAX do quadro
      modo="nao_rotulados"  - recortes que ainda estão em nao_rotulado
      modo="tudo"           - todos os recortes, o exemplos.csv e o registro de origem
    Devolve quantos recortes foram apagados.
    """
    caminho_csv = os.path.join(pasta_dados, "exemplos.csv")
    dados = _ler_csv(pasta_dados)

    if modo == "tudo":
        apagar = {p: _recortes(os.path.join(pasta_dados, p)) for p in PASTAS}
        for nome in ("exemplos.csv", "origem_rotulos.csv"):
            if os.path.exists(os.path.join(pasta_dados, nome)):
                os.remove(os.path.join(pasta_dados, nome))
        dados = None
    elif modo == "nao_rotulados":
        apagar = {"nao_rotulado": pendentes(pasta_dados)}
    else:
        grandes = set() if dados is None else set(dados.loc[dados["area_rel"] > cfg.AREA_MAX, "arquivo"])
        apagar = {p: [a for a in _recortes(os.path.join(pasta_dados, p)) if a in grandes] for p in PASTAS}

    apagados = 0
    for pasta, arquivos in apagar.items():
        for a in arquivos:
            os.remove(os.path.join(pasta_dados, pasta, a))
            apagados += 1

    # Tira do CSV as linhas cujo recorte não existe mais em nenhuma pasta.
    if dados is not None:
        existentes = {a for p in PASTAS for a in _recortes(os.path.join(pasta_dados, p))}
        dados[dados["arquivo"].isin(existentes)].to_csv(caminho_csv, index=False)
    return apagados
