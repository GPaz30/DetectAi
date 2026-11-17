"""
DetectAI - Detector de Movimento
================================
Disciplina: Algoritmos de Processamento de Imagem

Este arquivo tem SÓ a interface (Streamlit). O trabalho está nos outros arquivos:

    config.py          parâmetros do projeto
    deteccao.py        etapas 2 a 9: do quadro às caixas de movimento
    reconhecimento.py  dá nome às caixas (YOLO ou modelo treinado pelo grupo)
    rastreamento.py    acompanha os objetos entre os quadros; caixas estáveis
    caracteristicas.py números que descrevem cada objeto para o modelo do grupo
    exemplos.py        exemplos de treino: coleta, rotulação e limpeza
    treinar_modelo.py  treino e avaliação do modelo do grupo
    processamento.py   junta tudo, quadro a quadro
    desenho.py         desenha as caixas e os contornos nas imagens

Como executar:
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, replace

import cv2
import streamlit as st
import streamlit.components.v1 as components

import config as cfg
import desenho
import exemplos
import reconhecimento
from deteccao import pre_processar, redimensionar
from processamento import Opcoes, informacoes_do_video, ler_quadro, processar_video

# O treino precisa do scikit-learn. Se ele não estiver instalado, o resto do
# programa continua funcionando, e a aba Treinamento avisa.
try:
    import treinar_modelo as tm
    ERRO_TREINO = None
except Exception as erro:
    tm, ERRO_TREINO = None, erro

TIPOS_DE_VIDEO = ["mp4", "avi", "mov", "mkv"]
MODO_YOLO, MODO_GRUPO = "Inteligente (YOLO)", "Treinado pelo grupo"
CLASSES_DE_ROTULO = ["pessoa", "veiculo", "animal", "outro", "descartar"]

TEXTO_ETAPAS = """
1. **Original:** o quadro como está no vídeo.
2. **Redimensionado:** reduzido para a largura escolhida; menos pixels, processamento mais rápido.
3. **Tons de cinza:** só a intensidade de luz (usado pela diferença de quadros e pelas características do modelo).
4. **Desfoque gaussiano:** suaviza o ruído da câmera, para ele não virar falso movimento.
5. **Subtração de fundo:** o MOG2 compara o quadro com o fundo aprendido; branco = movimento, cinza = sombra.
   Na diferença de quadros, compara com o quadro anterior.
6. **Sombras detectadas:** as marcadas pelo MOG2 mais as encontradas em HSV (mais escuro que o fundo,
   mesma cor). Código do grupo, sempre ativo.
7. **Máscara sem sombras:** só o movimento real.
8. **Limpeza morfológica:** a mediana tira pontinhos; a abertura remove manchas pequenas; o fechamento tapa buracos.
9. **Contornos:** cada mancha branca vira um contorno; os menores que o tamanho mínimo são descartados
   (vermelho), as caixas próximas são unidas e os objetos grudados são separados por watershed
   (linhas rosa); as caixas finais aparecem em laranja.
10. **Resultado:** cada caixa recebe o nome dado pelo YOLO ou pelo modelo do grupo, com votação entre quadros.
   Cores: verde = pessoas, azul = veículos, laranja = animais, cinza = outros.
"""


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------
def configurar_pagina():
    """Título da aba, largura da página, CSS e idioma."""
    st.set_page_config(page_title="DetectAI", layout="wide")

    # CSS:
    #  - esconde tudo o que o Streamlit põe no canto superior direito: o aviso de
    #    execução ("Running..."), o menu e o botão Deploy. Para interromper um
    #    processamento há o botão "Parar" na própria página;
    #  - espaço no topo, para o título não ficar atrás da barra do Streamlit;
    #  - largura máxima da área principal, para as imagens não ficarem gigantes.
    st.markdown(
        "<style>"
        '[data-testid="stStatusWidget"],[data-testid="stMainMenu"],[data-testid="stAppDeployButton"],'
        '[data-testid="stToolbarActions"],#MainMenu,.stDeployButton{display:none!important}'
        ".block-container{padding-top:3.5rem;max-width:1500px}"
        "</style>",
        unsafe_allow_html=True,
    )

    # O Streamlit declara a página como inglês, então o Chrome oferece traduzir e
    # "traduz" palavras que já estão em português. Aqui a página é declarada como
    # pt-BR, com o pedido para o navegador não traduzir.
    components.html(
        """
        <script>
          const doc = window.parent.document;
          doc.documentElement.lang = "pt-BR";
          doc.documentElement.setAttribute("translate", "no");
          doc.body.classList.add("notranslate");
          if (!doc.querySelector('meta[name="google"]')) {
            const m = doc.createElement("meta");
            m.name = "google"; m.content = "notranslate";
            doc.head.appendChild(m);
          }
        </script>
        """,
        height=0,
    )


def mostrar(destino, imagem, legenda, largura_max=None):
    """
    Mostra uma imagem do OpenCV ocupando a largura da coluna, sem passar dela.

    A própria função gera o JPEG, com qualidade QUALIDADE_JPEG: o JPEG de
    qualidade máxima do Streamlit é cerca de 6 vezes mais pesado e deixa o vídeo
    lento. 'largura_max' reduz a imagem antes de enviar (colunas menores).
    """
    if largura_max and imagem.shape[1] > largura_max:
        imagem = redimensionar(imagem, largura_max)
    _, jpg = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, cfg.QUALIDADE_JPEG])
    try:
        destino.image(jpg.tobytes(), caption=legenda, width="stretch")               # Streamlit novo
    except Exception:
        destino.image(jpg.tobytes(), caption=legenda, use_container_width=True)      # Streamlit antigo


# ----------------------------------------------------------------------
# Modelos e vídeo (carregados uma vez só)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def modelo_yolo():
    """Carrega o YOLO uma única vez (o cache evita recarregar a cada clique)."""
    return reconhecimento.carregar_yolo()


@st.cache_resource(show_spinner=False)
def _modelo_grupo(data_do_arquivo):
    """'data_do_arquivo' faz o cache recarregar o modelo depois de um novo treino."""
    return reconhecimento.carregar_modelo_grupo()


def modelo_grupo():
    """O modelo treinado pelo grupo, ou None se ainda não houver um."""
    if not os.path.exists(cfg.ARQ_MODELO):
        return None
    return _modelo_grupo(os.path.getmtime(cfg.ARQ_MODELO))


def salvar_temporario(arquivo):
    """Grava um vídeo enviado num arquivo temporário (o OpenCV precisa de um caminho)."""
    sufixo = os.path.splitext(arquivo.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
        tmp.write(arquivo.getvalue())
    return tmp.name


def caminho_do_video(arquivo):
    """Caminho do vídeo da barra lateral. Ele só é gravado de novo quando o vídeo muda."""
    chave = (arquivo.name, arquivo.size)
    salvo = st.session_state.get("video")
    if salvo and salvo[0] == chave and os.path.exists(salvo[1]):
        return salvo[1]
    if salvo and os.path.exists(salvo[1]):
        os.remove(salvo[1])                       # apaga o vídeo anterior
    caminho = salvar_temporario(arquivo)
    st.session_state["video"] = (chave, caminho)
    return caminho


# ----------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------
@dataclass
class Controles:
    """O que foi escolhido na barra lateral."""
    arquivo: object      # vídeo enviado (None se não houver)
    usar_yolo: bool      # True = Inteligente (YOLO); False = Treinado pelo grupo
    grupos: list         # grupos que aparecem no resultado
    sensibilidade: int
    area_min: int
    usa_mog: bool        # True = MOG2; False = diferença de quadros
    coletar: bool        # salvar exemplos de treino durante o processamento


def barra_lateral():
    with st.sidebar:
        st.subheader("Entrada")
        arquivo = st.file_uploader("Vídeo", type=TIPOS_DE_VIDEO)

        st.subheader("Detecção")
        # YOLO = modelo externo pronto; "Treinado pelo grupo" = modelo da aba Treinamento.
        modo = st.radio("Reconhecimento", [MODO_YOLO, MODO_GRUPO])
        grupos = st.multiselect("Mostrar", list(cfg.GRUPOS), default=list(cfg.GRUPOS[:3]))
        # Maior = detecta movimentos mais sutis (e mais ruído).
        sensibilidade = st.slider("Sensibilidade ao movimento", 1, 10, 5)
        # Regiões menores que isso são ignoradas.
        area_min = st.slider("Tamanho mínimo (px²)", 100, 8000, 400, 100)
        # A comparação entre os dois métodos vai no relatório.
        metodo = st.radio("Método de fundo", ["MOG2", "Diferença de quadros"])
        # Salva recortes dos objetos para o grupo rotular e treinar.
        coletar = st.checkbox("Coletar exemplos para treino", value=False)
    return Controles(arquivo, modo == MODO_YOLO, grupos, sensibilidade, area_min, metodo == "MOG2", coletar)


# ----------------------------------------------------------------------
# Aba Processamento
# ----------------------------------------------------------------------
class Tela:
    """Os espaços das 10 etapas, na ordem em que acontecem, e o que aparece em cada um."""

    def __init__(self, opcoes):
        self.usa_mog = opcoes.usa_mog
        st.markdown("**Etapa 2 · Pré-processamento**")
        c = st.columns(4)
        self.espacos = {"original": c[0].empty(), "redim": c[1].empty(), "cinza": c[2].empty(),
                        "suave": c[3].empty()}
        st.markdown("**Etapa 3 · Segmentação**")
        c = st.columns(4)
        self.espacos.update({"fundo": c[0].empty(), "sombras": c[1].empty(), "sem_sombras": c[2].empty(),
                             "limpa": c[3].empty()})
        st.markdown("**Detecção e resultado**")
        c = st.columns(cfg.PROPORCAO_FINAL, gap="medium")
        self.espacos.update({"contornos": c[0].empty(), "resultado": c[1].empty()})

        self.legendas = {
            "original": "1. Original",
            "redim": "2. Redimensionado",
            "cinza": "3. Tons de cinza",
            "suave": f"4. Desfoque gaussiano (kernel {opcoes.desfoque})",
            "fundo": "5. Subtração de fundo: MOG2 (branco = movimento, cinza = sombra)" if self.usa_mog
                     else "5. Diferença de quadros",
            "sombras": "6. Sombras detectadas (MOG2 + HSV)",
            "sem_sombras": "7. Máscara sem sombras",
            "limpa": "8. Limpeza morfológica (mediana, abertura, fechamento)",
            "contornos": "9. Contornos (verde = aceito, vermelho = pequeno), divisões (rosa) e caixas",
            "resultado": "10. Resultado",
        }

    def _mostrar(self, imagens):
        for etapa, imagem in imagens.items():
            mostrar(self.espacos[etapa], imagem, self.legendas[etapa],
                    largura_max=None if etapa == "resultado" else cfg.LARG_ETAPA)

    def pre_visualizar(self, quadro, opcoes):
        """Antes de processar: o quadro escolhido passa só pelo pré-processamento."""
        redim, cinza, suave, _ = pre_processar(quadro, opcoes.largura, opcoes.desfoque)
        espera = desenho.imagem_aviso("aparece ao processar o video")
        so_mog = espera if self.usa_mog else desenho.imagem_aviso("nao usado na diferenca de quadros")
        self._mostrar({"original": quadro, "redim": redim, "cinza": cinza, "suave": suave, "fundo": espera,
                       "sombras": so_mog, "sem_sombras": so_mog, "limpa": espera, "contornos": espera,
                       "resultado": redim})

    def mostrar_resultado(self, r, area_min, etapas=True):
        """
        Durante o processamento: as 10 etapas do quadro 'r'.
        etapas=False atualiza só o Resultado (etapa 10): as outras 9 imagens são
        atualizadas menos vezes, para o navegador não ficar sobrecarregado.
        """
        imagens = {"resultado": desenho.desenhar_objetos(r.redim, r.objetos)}
        if etapas:
            imagens.update(original=r.original, redim=r.redim, cinza=r.cinza, suave=r.suave,
                           fundo=r.fundo, limpa=r.limpa,
                           contornos=desenho.desenhar_contornos(r.limpa, r.caixas, area_min, r.cortes))
            if self.usa_mog:
                imagens.update(sombras=r.sombras, sem_sombras=r.sem_sombras)
        self._mostrar(imagens)


def preparar_reconhecimento(usar_yolo, grupos, estado):
    """
    Carrega o modelo que vai dar nome aos objetos.
    Devolve (yolo, modelo do grupo, grupos a exibir).
    """
    yolo = None
    if usar_yolo:
        estado.markdown("Carregando o modelo YOLO...")
        try:
            yolo = modelo_yolo()
        except Exception as e:      # YOLO não instalado ou sem internet para baixar
            st.warning(f"YOLO indisponível; usando o modelo treinado pelo grupo. Detalhe: {e}")

    pacote, grupos = None, list(grupos)
    if yolo is None:
        pacote = modelo_grupo()
        if pacote is None:
            st.warning('Ainda não há modelo treinado pelo grupo: as regiões aparecem como "Outro". '
                       "Treine um modelo na aba Treinamento.")
            if "Outros" not in grupos:
                grupos.append("Outros")
    return yolo, pacote, grupos


def executar_processamento(caminho, controles, opcoes, tela, barra, estado):
    """Processa o vídeo inteiro, atualizando a tela, e mostra o resumo no final."""
    yolo, pacote, grupos = preparar_reconhecimento(controles.usar_yolo, controles.grupos, estado)
    opcoes = replace(opcoes, grupos=tuple(grupos))
    coletor = exemplos.Coletor(controles.arquivo.name) if controles.coletar else None

    def atualizar_tela(r, etapas=True):
        tela.mostrar_resultado(r, opcoes.area_min, etapas)
        contagem = r.contagem()
        barra.progress(min(r.n / max(r.total, 1), 1.0))
        estado.markdown(
            f"**Quadro {r.n}/{r.total}**  ·  Pessoas **{contagem['Pessoas']}**  ·  "
            f"Veículos **{contagem['Veículos']}**  ·  Animais **{contagem['Animais']}**"
            + ("  ·  _mudança de luz: quadro ignorado_" if r.mudanca_de_luz else "")
        )

    historico = {g: [] for g in cfg.GRUPOS}     # objetos de cada grupo, quadro a quadro
    analisados = ignorados = pulados = 0
    ultimo, ultima_tela, ultimas_etapas = None, 0.0, 0.0
    # Na tela o vídeo anda na velocidade real (pulando quadros se o computador não
    # acompanhar). Na coleta, não: todos os quadros viram exemplos.
    tempo_real = cfg.TEMPO_REAL and coletor is None
    try:
        for r in processar_video(caminho, opcoes, yolo, pacote, coletor, tempo_real):
            analisados += 1
            ignorados += r.mudanca_de_luz
            pulados += r.passos - 1
            for g, n in r.contagem().items():
                historico[g].append(n)
            ultimo = r
            # Enviar imagens ao navegador é a parte mais lenta. O Resultado é
            # atualizado no máximo FPS_TELA vezes por segundo, e as outras 9 etapas
            # no máximo FPS_ETAPAS vezes; o processamento não espera a tela.
            agora = time.time()
            if agora - ultima_tela >= 1.0 / cfg.FPS_TELA:
                etapas = agora - ultimas_etapas >= 1.0 / cfg.FPS_ETAPAS
                atualizar_tela(r, etapas)
                ultima_tela = agora
                if etapas:
                    ultimas_etapas = agora
    finally:
        if coletor is not None:
            coletor.fechar()

    if ultimo is not None:
        atualizar_tela(ultimo)                   # garante que o último quadro apareça
    barra.progress(1.0)

    st.success(f"Concluído: {analisados} quadros analisados."
               + (f" {ignorados} ignorados por mudança de luz." if ignorados else "")
               + (f" {pulados} pulados para o vídeo andar na velocidade real." if pulados else ""))
    if coletor is not None:
        st.info(f"{coletor.total} exemplos coletados. Rotule-os na aba Treinamento.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Máx. pessoas simultâneas", max(historico["Pessoas"], default=0))
    c2.metric("Máx. veículos simultâneos", max(historico["Veículos"], default=0))
    c3.metric("Máx. animais simultâneos", max(historico["Animais"], default=0))
    grafico = {g: v for g, v in historico.items() if g in grupos and any(v)}
    if grafico:
        st.line_chart(grafico)


def aba_processamento(controles, opcoes, coluna_do_quadro):
    """Pré-visualização, botões e as 10 etapas do processamento."""
    if controles.arquivo is None:
        st.info("Carregue um vídeo na barra lateral para começar.")
        return
    caminho = caminho_do_video(controles.arquivo)
    video = informacoes_do_video(caminho)
    if video is None:
        st.error("Não foi possível abrir o vídeo. Tente converter para .mp4.")
        return

    total = video["quadros"]
    indice = coluna_do_quadro.slider("Quadro de pré-visualização", 0, max(total - 1, 1), 0) if total > 1 else 0
    st.caption(f"{video['largura']}x{video['altura']} px  ·  {video['fps']:.1f} FPS  ·  {total} quadros")

    b1, b2, _ = st.columns([1, 1, 5])
    executar = b1.button("Processar vídeo", type="primary")
    # Qualquer clique durante o processamento faz o Streamlit recomeçar a página,
    # o que interrompe a análise; este botão existe só para isso.
    b2.button("Parar", help="Interrompe o processamento em andamento.")
    barra = st.progress(0.0)
    estado = st.empty()

    tela = Tela(opcoes)
    quadro = ler_quadro(caminho, indice)
    if quadro is not None:
        tela.pre_visualizar(quadro, opcoes)
    if executar:
        executar_processamento(caminho, controles, opcoes, tela, barra, estado)

    with st.expander("O que cada etapa faz"):
        st.markdown(TEXTO_ETAPAS)


# ----------------------------------------------------------------------
# Aba Treinamento
# ----------------------------------------------------------------------
def treino_automatico(videos, opcoes, certeza_min, sem_sugestao_vira_outro):
    """
    Treina o modelo do grupo sozinho, com o YOLO como "professor":
      1) coleta os exemplos dos vídeos com o YOLO, sem exibir nada (mais rápido);
      2) rotula os exemplos pelas sugestões do YOLO;
      3) treina e avalia o modelo do grupo.
    """
    estado, barra = st.empty(), st.progress(0.0)
    estado.markdown("Carregando o modelo YOLO...")
    try:
        yolo = modelo_yolo()
    except Exception as e:
        estado.empty()
        st.error(f"O treino automático precisa do YOLO, que não pôde ser carregado: {e}")
        return

    # 1) Coleta. Vídeos que já têm exemplos são pulados, para não repetir exemplos.
    ja_coletados = set(exemplos.videos_coletados())
    novos = [v for v in videos if v.name not in ja_coletados]
    opcoes = replace(opcoes, grupos=cfg.GRUPOS)
    coletados = 0
    for k, video in enumerate(novos):
        caminho = salvar_temporario(video)
        coletor = exemplos.Coletor(video.name)
        completo = False
        try:
            ultima = 0.0
            for r in processar_video(caminho, opcoes, yolo=yolo, coletor=coletor):
                if time.time() - ultima >= 0.25:
                    barra.progress(min((k + r.n / max(r.total, 1)) / len(novos), 1.0))
                    estado.markdown(f"Coletando exemplos: vídeo {k + 1} de {len(novos)} "
                                    f"(**{video.name}**), quadro {r.n}/{r.total}")
                    ultima = time.time()
            completo = True
            coletados += coletor.total
        finally:
            coletor.fechar()
            os.remove(caminho)
            if not completo:      # interrompido (botão Parar): não deixa o vídeo pela metade
                exemplos.apagar_coleta(coletor.prefixo)

    # 2) Rotulação automática e 3) treino.
    estado.markdown("Rotulando os exemplos com as sugestões do YOLO...")
    rotulos = exemplos.rotular_automatico(certeza_min, sem_sugestao_vira_outro)
    estado.markdown("Treinando o modelo do grupo...")
    try:
        st.session_state["resultado_treino"] = tm.treinar("auto")
        erro = None
    except Exception as e:
        erro = str(e)
    barra.progress(1.0)
    estado.empty()
    st.session_state.pop("ultimo_rotulo", None)
    st.session_state["resumo_automatico"] = {"videos": len(novos), "pulados": len(videos) - len(novos),
                                              "coletados": coletados, "rotulos": rotulos, "erro": erro}


def secao_treino_automatico(opcoes):
    st.subheader("Treino automático pelo YOLO")
    st.markdown("Envie os vídeos e clique no botão: o programa **coleta** os exemplos com o YOLO, **rotula** "
                "sozinho pelas sugestões dele e **treina** o modelo do grupo. Um recorte só é aceito se o YOLO "
                "tiver a certeza mínima **e** não mudar de ideia sobre o objeto ao longo do vídeo; os outros "
                "são descartados. Use vídeos variados; com dois ou mais, parte deles fica reservada para "
                "medir a acurácia.")
    videos = st.file_uploader("Vídeos para o treino automático", type=TIPOS_DE_VIDEO,
                              accept_multiple_files=True, key="videos_treino") or []
    c1, c2 = st.columns(2)
    certeza = c1.slider("Certeza mínima do YOLO para aceitar um rótulo (%)", 30, 95, cfg.CERTEZA_AUTOMATICA, 5,
                        help="Recortes abaixo disso são descartados. Maior = menos erros, mas menos exemplos.")
    vira_outro = c2.checkbox('Movimento que o YOLO não reconheceu vira "Outro"', value=True,
                             help="Em geral é sombra, reflexo ou ruído; mas pode ser um objeto que o YOLO "
                                  "não enxergou.")
    n_pendentes = len(exemplos.pendentes())
    if n_pendentes:
        st.caption(f"Os {n_pendentes} recortes que já esperam rótulo também serão rotulados pelo YOLO.")
    b1, b2, _ = st.columns([2, 1, 4])
    treinar = b1.button("Treinar com o YOLO", type="primary", disabled=not videos and not n_pendentes)
    b2.button("Parar", key="parar_treino", help="Interrompe o treino automático em andamento.")
    if treinar and (videos or n_pendentes):
        treino_automatico(videos, opcoes, certeza / 100, bool(vira_outro))

    resumo = st.session_state.get("resumo_automatico")
    if resumo:
        rotulos = resumo["rotulos"]
        partes = [f"{cfg.NOME_DA_PASTA[p]}: {rotulos[p]}" for p in ("pessoa", "veiculo", "animal", "outro")
                  if rotulos.get(p)]
        texto = (f"Treino automático: {resumo['coletados']} recortes coletados de {resumo['videos']} vídeo(s). "
                 f"Rotulados pelo YOLO: {', '.join(partes) or 'nenhum'}; "
                 f"descartados por dúvida: {rotulos.get('descartar', 0)}.")
        if resumo["pulados"]:
            texto += f" {resumo['pulados']} vídeo(s) já tinham exemplos e foram pulados."
        if rotulos.get("sem_yolo"):
            texto += f" {rotulos['sem_yolo']} recorte(s) coletados sem o YOLO ficaram para rotular à mão."
        st.success(texto)
        if resumo["erro"]:
            st.error(f"Os exemplos foram rotulados, mas não foi possível treinar: {resumo['erro']}")
        st.caption("Os rótulos vieram do YOLO, sem revisão: o modelo do grupo aprende também os erros dele. "
                   "Declare isso no relatório e, se puder, confira as pastas de dataset.")


def rotular_e_recarregar(movimentos, origem):
    """Aplica uma rotulação, guarda-a para o botão Desfazer e recarrega a página."""
    exemplos.rotular(movimentos, origem)
    st.session_state["ultimo_rotulo"] = movimentos
    st.rerun()


def secao_rotular():
    st.subheader("2. Rotular")
    contagem = exemplos.contar_rotulos()
    for coluna, (titulo, pasta) in zip(st.columns(6), [("Pessoa", "pessoa"), ("Veículo", "veiculo"),
                                                       ("Animal", "animal"), ("Outro", "outro"),
                                                       ("Descartados", "descartar"), ("A rotular", "nao_rotulado")]):
        coluna.metric(titulo, contagem[pasta])

    fila = exemplos.pendentes()
    if not fila:
        st.info("Nenhum recorte para rotular. Colete exemplos processando vídeos com a coleta ligada.")
        return
    sugestoes, _ = exemplos.carregar_sugestoes()
    pasta_fila = os.path.join(cfg.PASTA_DADOS, "nao_rotulado")

    # ---- Aceitar em lote as sugestões do YOLO ----
    if any(a in sugestoes for a in fila):
        with st.expander("Rotulação assistida pelo YOLO: aceitar sugestões em lote", expanded=True):
            st.caption("O YOLO sugeriu um rótulo para os recortes coletados no modo Inteligente. Aceite de uma "
                       "vez os objetos em que ele tem muita certeza e revise o resto um por um, abaixo. "
                       "O YOLO não conhece a classe Outro (sombras e ruído): essa continua com vocês.")
            # Quanto maior, menos objetos entram no lote, mas com menos erros.
            limite = st.slider("Certeza mínima do YOLO (%)", 50, 99, 80, 1)
            lote = exemplos.sugestoes_em_lote(limite / 100)
            por_classe = ", ".join(f"{cfg.NOME_DA_PASTA[p]}: {sum(1 for _, d in lote if d == p)}"
                                   for p in ("pessoa", "veiculo", "animal") if any(d == p for _, d in lote))
            st.caption(f"Com {limite}% ou mais: {len(lote)} objetos ({sum(len(a) for a, _ in lote)} recortes)"
                       + (f" — {por_classe}." if por_classe else "."))
            if st.button(f"Aceitar as {len(lote)} sugestões", disabled=not lote) and lote:
                rotular_e_recarregar(lote, "yolo_lote")

    # ---- Um objeto por vez ----
    atual = fila[0]
    do_objeto = sorted(a for a in fila if exemplos.objeto_de(a) == exemplos.objeto_de(atual))
    objeto_inteiro = st.toggle("Rotular o objeto inteiro de uma vez", value=True,
                               help="Um clique dá o mesmo rótulo a todos os recortes deste objeto "
                                    "(o rastreador já sabe que são o mesmo). Desligue para rotular um por um.")
    alvo = do_objeto if objeto_inteiro else [atual]
    st.caption(f"{len(fila)} recortes de {len(exemplos.agrupar_por_objeto(fila))} objetos para rotular.")

    # Até 6 recortes do objeto, para conferir se são todos da mesma coisa.
    amostra = alvo if len(alvo) <= 6 else [alvo[int(k * (len(alvo) - 1) / 5)] for k in range(6)]
    for coluna, arquivo in zip(st.columns(6), amostra):
        imagem = cv2.imread(os.path.join(pasta_fila, arquivo))
        if imagem is not None:
            mostrar(coluna, imagem, None, largura_max=200)

    # Sugestão do YOLO para este objeto, com botão de confirmar.
    sugerida, certeza = exemplos.sugestao_do_objeto(alvo, sugestoes)
    if sugerida:
        c_texto, c_botao = st.columns([3, 2])
        c_texto.info(f"O YOLO sugere: **{cfg.NOME_DA_PASTA[sugerida]}** ({certeza:.0%} de certeza). "
                     "Confira as imagens.")
        if c_botao.button(f"Confirmar: {cfg.NOME_DA_PASTA[sugerida]}", type="primary", key="confirmar_sugestao"):
            rotular_e_recarregar([(alvo, sugerida)], "yolo_confirmado")
    elif sugestoes:
        st.caption("O YOLO não reconheceu este objeto: pode ser **Outro** (sombra, ruído) ou algo que ele "
                   "não enxergou.")

    plural = "s" if len(alvo) > 1 else ""
    st.markdown(f"**O que está no retângulo verde?** ({len(alvo)} recorte{plural} "
                f"recebe{'m' if plural else ''} este rótulo)")
    st.caption("**Outro** = sombra, reflexo, folha, ruído. **Descartar** = recorte ruim ou duvidoso "
               "(não entra no treino). Se os recortes mostram coisas diferentes, desligue a opção acima.")
    for coluna, pasta in zip(st.columns(5), CLASSES_DE_ROTULO):
        if coluna.button(cfg.NOME_DA_PASTA[pasta], key=f"rotulo_{pasta}"):
            # Clicar na classe que o YOLO sugeriu também é uma confirmação.
            rotular_e_recarregar([(alvo, pasta)], "yolo_confirmado" if pasta == sugerida else "manual")

    ultimo = st.session_state.get("ultimo_rotulo")
    if ultimo:
        destinos = ", ".join(sorted({d for _, d in ultimo}))
        if st.button(f"Desfazer último ({sum(len(a) for a, _ in ultimo)} recorte(s) em {destinos})"):
            exemplos.desfazer(ultimo)
            st.session_state.pop("ultimo_rotulo")
            st.rerun()
    st.caption("Também dá para rotular pelo explorador de arquivos: mova os recortes de "
               "dataset/nao_rotulado para dataset/pessoa, veiculo, animal ou outro.")


def secao_limpar():
    def limpar(modo, mensagem):
        n = exemplos.limpar_exemplos(modo)
        st.session_state.pop("ultimo_rotulo", None)
        st.session_state["mensagem_limpeza"] = mensagem.format(n=n)
        st.rerun()

    with st.expander("Limpar exemplos salvos"):
        st.caption("Os recortes e o exemplos.csv ficam na pasta dataset. Apagar não tem como desfazer.")
        c1, c2 = st.columns(2)
        if c1.button(f"Remover exemplos com caixas grandes (mais de {cfg.AREA_MAX:.0%} do quadro)"):
            limpar("grandes", "{n} exemplos com caixas grandes removidos.")
        if c2.button("Apagar recortes ainda não rotulados"):
            limpar("nao_rotulados", "{n} recortes não rotulados apagados.")
        certeza = st.checkbox("Quero apagar TODOS os exemplos (rotulados ou não) e o modelo treinado")
        # O botão só funciona com a confirmação marcada (conferida de novo aqui, por segurança).
        if st.button("Apagar tudo e recomeçar", disabled=not certeza) and certeza:
            shutil.rmtree(cfg.PASTA_MODELO, ignore_errors=True)
            for chave in ("resultado_treino", "resumo_automatico"):
                st.session_state.pop(chave, None)
            limpar("tudo", "{n} exemplos e o modelo foram apagados.")
        if st.session_state.get("mensagem_limpeza"):
            st.success(st.session_state.pop("mensagem_limpeza"))


def secao_treinar():
    st.subheader("3. Treinar e avaliar")
    tipos = {"Automático (testa todos e escolhe o melhor)": "auto", "Floresta aleatória": "floresta",
             "Gradient boosting": "boosting", "k vizinhos mais próximos (k-NN)": "knn",
             "Árvore de decisão (mais fácil de explicar)": "arvore"}
    c1, c2 = st.columns(2)
    tipo = c1.radio("Tipo de modelo", list(tipos),
                    help="Automático: testa os quatro com validação cruzada nos vídeos de treino e fica com o "
                         "de maior nota. A árvore usa só as 10 características e mostra as regras que aprendeu; "
                         "os outros também usam a forma do objeto (HOG).")
    teste = c2.multiselect("Vídeos reservados para teste", exemplos.videos_coletados(),
                           help="Vazio = o programa escolhe ~25% dos vídeos. Esses vídeos NÃO são "
                                "usados para aprender, só para medir a acurácia.")
    if st.button("Treinar modelo", type="primary"):
        try:
            with st.spinner("Treinando..."):
                st.session_state["resultado_treino"] = tm.treinar(tipos[tipo], teste or None)
        except Exception as e:
            st.error(f"Não foi possível treinar: {e}")


def mostrar_resultado_do_treino():
    r = st.session_state.get("resultado_treino")
    if not r:
        return
    if r["aviso"]:
        st.warning(r["aviso"])
    st.markdown(f"**Modelo escolhido: {r['nome']}**" + (" (escolha automática)" if r["automatico"] else ""))
    origem = r.get("origem", {})
    if any(o.startswith("yolo") for o in origem):
        st.caption("Origem dos rótulos: " + ", ".join(f"{tm.NOMES_ORIGENS.get(o, o)}: {n}" for o, n in origem.items())
                   + ". Declare no relatório que o YOLO ajudou a rotular.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Acurácia do modelo (teste)", f"{r['acuracia']:.0%}")
    m2.metric("Regras manuais antigas (mesmo teste)", f"{r['acuracia_regra']:.0%}")
    m3.metric("Exemplos (treino / teste)", f"{r['n_treino']} / {r['n_teste']}")

    if len(r["comparacao"]) > 1:
        def pct(v):
            return "-" if v != v else f"{v:.0%}"     # v != v só é verdade para NaN
        st.caption("Comparação dos modelos. A escolha usa a validação nos vídeos de treino; "
                   "as colunas de teste mostram como cada um se saiu nos vídeos reservados.")
        st.table([{"Modelo": c["nome"] + (" (escolhido)" if c["tipo"] == r["tipo"] else ""),
                   "Validação (treino)": pct(c["cv"]), "Acurácia (teste)": pct(c["acuracia"]),
                   "Acurácia balanceada (teste)": pct(c["acuracia_bal"])} for c in r["comparacao"]])

    figuras = [f for f in r.get("figuras", []) if os.path.exists(f)]
    for coluna, figura in zip(st.columns(max(1, len(figuras))), figuras):
        coluna.image(figura)
    relatorio = os.path.join(cfg.PASTA_MODELO, "relatorio_treino.txt")
    if os.path.exists(relatorio):
        with st.expander("Relatório completo (também salvo em modelo/relatorio_treino.txt)"):
            st.code(open(relatorio, encoding="utf-8").read())


def aba_treinamento(opcoes):
    """Treinar o modelo do grupo: de forma automática pelo YOLO, ou passo a passo."""
    if tm is None:
        st.warning("Para treinar, instale o scikit-learn: pip install -r requirements.txt  "
                   f"(detalhe: {ERRO_TREINO})")
        return

    secao_treino_automatico(opcoes)

    st.markdown("---")
    st.markdown("### Treino passo a passo")
    st.subheader("1. Coletar exemplos")
    st.markdown("Marque **Coletar exemplos para treino** na barra lateral e processe vídeos na aba "
                "**Processamento**. Cada objeto em movimento vira um recorte para rotular. Use vídeos "
                "variados e guarde alguns **só para teste**.")
    secao_rotular()
    secao_limpar()
    secao_treinar()

    mostrar_resultado_do_treino()
    pacote = modelo_grupo()
    if pacote:
        st.success(f"Modelo em uso: {tm.NOMES_MODELOS.get(pacote['tipo'], pacote['tipo']).lower()}, "
                   f"treinado em {pacote['data']} com {pacote['n_exemplos']} exemplos "
                   f"(acurácia no teste: {pacote['acuracia_teste']:.0%}).")
    else:
        st.info("Ainda não há modelo treinado.")


# ----------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------
configurar_pagina()
st.title("DetectAI")
st.caption("Detector de movimento em vídeos de câmera fixa")

controles = barra_lateral()
aba_proc, aba_treino = st.tabs(["Processamento", "Treinamento"])

# Parâmetros do pré-processamento: valem para a pré-visualização, para a
# detecção e para a coleta do treino automático.
with aba_proc:
    c_largura, c_desfoque, c_quadro = st.columns(3)
    opcoes = Opcoes(
        largura=c_largura.slider("Largura do quadro (px)", 320, 960, cfg.LARGURA_PADRAO, 40),
        desfoque=c_desfoque.slider("Desfoque gaussiano (kernel)", 1, 15, cfg.DESFOQUE_PADRAO, 2),
        sensibilidade=controles.sensibilidade,
        area_min=controles.area_min,
        usa_mog=controles.usa_mog,
    )

with aba_treino:        # funciona mesmo sem vídeo carregado na barra lateral
    aba_treinamento(opcoes)

with aba_proc:
    aba_processamento(controles, opcoes, c_quadro)
