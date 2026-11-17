"""
DetectAI - Treinamento do modelo do grupo
=========================================
Treina o classificador que dá nome às regiões em movimento
(Pessoa, Veiculo, Animal, Outro) a partir dos exemplos que o grupo coletou
e rotulou.

O que o modelo recebe de cada exemplo:
  - 10 características geométricas, de movimento e de cor (caracteristicas.py);
  - (exceto na árvore) 144 números do descritor HOG, que descrevem a FORMA do
    objeto dentro da caixa. O HOG é calculado a partir dos recortes salvos.

Modelos disponíveis:
  arvore    - Árvore de decisão: regras simples, fácil de explicar (só as 10 características)
  floresta  - Floresta aleatória: 300 árvores votando; mais precisa e estável
  boosting  - Gradient boosting: árvores em sequência, cada uma corrige os erros da anterior
  knn       - k vizinhos mais próximos: classe dos exemplos mais parecidos
  auto      - testa todos com validação cruzada nos vídeos de treino e escolhe o melhor

Pastas usadas (dentro da pasta do projeto):
    dataset/                os exemplos e seus rótulos (ver exemplos.py)
    modelo/
      modelo_grupo.joblib   o modelo treinado (usado pelo app)
      relatorio_treino.txt  resultados: comparação dos modelos, matriz de confusão...
      matriz_confusao.png   figura da matriz de confusão
      arvore.png            figura da árvore (se a árvore for o modelo escolhido)
      importancia.png       importância das características (floresta)

Uso pelo terminal (com o .venv ativo):
    python treinar_modelo.py                         (automático, teste escolhido sozinho)
    python treinar_modelo.py --modelo floresta       (um modelo específico)
    python treinar_modelo.py --teste rua2.mp4        (escolhe os vídeos de teste)
Também pode ser chamado pela aba "Treinamento" do app.
"""
import argparse
import datetime
import os
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import caracteristicas as car
import config as cfg
import exemplos

# Aviso inofensivo quando o teste não tem exemplos de alguma classe (ex.: nenhum animal).
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

NOMES_MODELOS = {
    "arvore": "Árvore de decisão",
    "floresta": "Floresta aleatória",
    "boosting": "Gradient boosting",
    "knn": "k vizinhos mais próximos (k-NN)",
}
# Como cada origem de rótulo aparece no relatório (ver exemplos.registrar_origem).
NOMES_ORIGENS = {
    "manual": "manuais",
    "yolo_confirmado": "sugestão do YOLO confirmada",
    "yolo_lote": "sugestão do YOLO aceita em lote",
    "yolo_automatico": "automático pelo YOLO (sem revisão)",
    "sem_registro": "sem registro",
}
# A árvore usa só as 10 características, para continuar fácil de explicar.
USA_HOG = {"arvore": False, "floresta": True, "boosting": True, "knn": True}


# ----------------------------------------------------------------------
# 1) Carregar os exemplos rotulados
# ----------------------------------------------------------------------
def carregar_exemplos(pasta_dados=cfg.PASTA_DADOS):
    """
    Junta as características (exemplos.csv) com o rótulo, que é o nome da
    pasta para onde o grupo moveu cada recorte. Recortes ainda em
    'nao_rotulado' ou em 'descartar' ficam de fora.
    """
    csv = os.path.join(pasta_dados, "exemplos.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError("Não há exemplos coletados ainda (dataset/exemplos.csv não existe).")
    dados = pd.read_csv(csv).drop_duplicates(subset="arquivo", keep="last")

    rotulos, caminhos = {}, {}
    for pasta_classe, nome in car.CLASSES.items():
        p = os.path.join(pasta_dados, pasta_classe)
        if os.path.isdir(p):
            for arq in os.listdir(p):
                rotulos[arq] = nome
                caminhos[arq] = os.path.join(p, arq)
    dados["classe"] = dados["arquivo"].map(rotulos)
    dados["caminho"] = dados["arquivo"].map(caminhos)
    return dados.dropna(subset=["classe"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2) Montar a tabela de números (X) que o modelo recebe
# ----------------------------------------------------------------------
def calcular_hog(dados):
    """
    HOG de cada exemplo, a partir do recorte salvo: localiza a caixa dentro do
    recorte (que tem margem) e descreve a forma do objeto.
    Calcula também a versão espelhada (usada para aumentar os dados de treino).
    """
    normal, espelhado = [], []
    for _, linha in dados.iterrows():
        img = cv2.imread(linha["caminho"])
        if img is None:
            normal.append(np.zeros(car.N_HOG, np.float32))
            espelhado.append(np.zeros(car.N_HOG, np.float32))
            continue
        caixa = car.caixa_no_recorte(int(linha["x"]), int(linha["y"]), int(linha["w"]), int(linha["h"]))
        regiao = car.recorte_caixa(img, caixa)
        normal.append(car.aparencia(regiao))
        espelhado.append(car.aparencia(regiao, espelhar=True))
    return np.array(normal), np.array(espelhado)


def montar_X(dados, hog, usa_hog):
    base = dados[car.NOMES_ATRIBUTOS].values.astype(np.float32)
    return np.hstack([base, hog]) if usa_hog else base


def aumentar(X, y, hog_esp, usa_hog):
    """
    Aumento de dados: cada exemplo de treino entra também espelhado na
    horizontal (uma pessoa andando para a esquerda vira uma andando para a
    direita). As 10 características não mudam com o espelho; só o HOG muda.
    Dobra a quantidade de exemplos sem precisar rotular mais nada.
    """
    if not usa_hog:
        return X, y
    X_esp = np.hstack([X[:, :len(car.NOMES_ATRIBUTOS)], hog_esp])
    return np.vstack([X, X_esp]), np.concatenate([y, y])


# ----------------------------------------------------------------------
# 3) Separar treino e teste
# ----------------------------------------------------------------------
def separar(dados, videos_teste=None):
    """
    Separa POR VÍDEO: os exemplos de um vídeo vão todos para o treino ou todos
    para o teste. Se o mesmo vídeo estivesse nos dois lados, o modelo seria
    testado em quadros quase iguais aos que viu, e a nota pareceria melhor
    do que realmente é.
    Devolve (índices de treino, índices de teste, aviso).
    """
    videos = sorted(dados["video"].unique())
    aviso = ""
    if videos_teste:
        teste_mask = dados["video"].isin(videos_teste).values
    elif len(videos) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        _, idx_teste = next(gss.split(dados, groups=dados["video"]))
        teste_mask = np.isin(np.arange(len(dados)), idx_teste)
    else:
        aviso = ("Só há exemplos de UM vídeo: o teste usa quadros do mesmo vídeo do treino, "
                 "então a acurácia fica otimista. Colete exemplos de mais vídeos.")
        estrat = dados["classe"] if dados["classe"].value_counts().min() >= 2 else None
        tr, te = train_test_split(np.arange(len(dados)), test_size=0.25, random_state=42, stratify=estrat)
        return np.sort(tr), np.sort(te), aviso

    tr, te = np.where(~teste_mask)[0], np.where(teste_mask)[0]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError("A separação deixou o treino ou o teste vazio. Escolha outros vídeos de teste.")
    faltando = set(dados["classe"]) - set(dados["classe"].iloc[tr])
    if faltando:
        aviso = f"Classes sem nenhum exemplo no treino: {', '.join(sorted(faltando))}."
    return tr, te, aviso


# ----------------------------------------------------------------------
# 4) Os modelos
# ----------------------------------------------------------------------
def criar_modelo(tipo, n_treino):
    if tipo == "floresta":
        # Floresta aleatória: 300 árvores, cada uma treinada com um sorteio dos
        # exemplos e das características; a classe final é o voto da maioria.
        # Erra menos que uma árvore só e lida bem com os 144 números do HOG.
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=42)
    if tipo == "boosting":
        # Gradient boosting: árvores pequenas em sequência; cada nova árvore
        # aprende a corrigir os erros das anteriores.
        # (Sem class_weight: com ele e o HOG, o treino ficava ~25x mais lento.)
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=42)
    if tipo == "knn":
        # k-NN: classe dos 5 exemplos mais parecidos. O PCA resume os 154 números
        # em até 30 (senão a "distância" fica dominada por ruído) e o
        # StandardScaler põe tudo na mesma escala.
        n_comp = max(2, min(30, n_treino - 1))
        return make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=42),
                             KNeighborsClassifier(n_neighbors=max(1, min(5, n_treino)), weights="distance"))
    # Árvore de decisão: perguntas do tipo "razao <= 0,9?" em sequência.
    # max_depth limita o tamanho (evita decorar o treino e fica fácil de explicar);
    # class_weight="balanced" compensa classes com poucos exemplos (ex.: animais).
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=42)


def ajustar(tipo, X, y, hog_esp):
    """Cria e treina um modelo (com aumento de dados quando ele usa o HOG)."""
    Xa, ya = aumentar(X, y, hog_esp, USA_HOG[tipo])
    m = criar_modelo(tipo, len(ya))
    m.fit(Xa, ya)                                   # <- aqui o modelo APRENDE
    return m


def validacao_cruzada(tipo, X, y, hog_esp, grupos):
    """
    Nota do modelo usando SÓ os vídeos de treino: o treino é dividido em partes
    (por vídeo, quando possível); cada parte é testada com um modelo treinado nas
    outras. A nota é a acurácia balanceada (média do acerto de cada classe), que
    não é enganada por uma classe com muito mais exemplos.
    """
    n_videos = len(set(grupos))
    if n_videos >= 3:
        divisoes = GroupKFold(n_splits=min(4, n_videos)).split(X, y, grupos)
    else:
        n = int(min(4, pd.Series(y).value_counts().min()))
        if n < 2:
            return float("nan")
        divisoes = StratifiedKFold(n_splits=n, shuffle=True, random_state=42).split(X, y)
    notas = []
    for tr, te in divisoes:
        if len(set(y[tr])) < 2:
            continue
        m = ajustar(tipo, X[tr], y[tr], hog_esp[tr])
        notas.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(notas)) if notas else float("nan")


# ----------------------------------------------------------------------
# 5) Treinar, avaliar e salvar
# ----------------------------------------------------------------------
def treinar(tipo="auto", videos_teste=None, pasta_dados=cfg.PASTA_DADOS, pasta_modelo=cfg.PASTA_MODELO):
    """Executa o treinamento completo e devolve um dicionário com os resultados."""
    dados = carregar_exemplos(pasta_dados)
    if len(dados) < 10:
        raise ValueError(f"Só há {len(dados)} exemplos rotulados. Rotule pelo menos uns 10 (o ideal é 100+ por classe).")
    if dados["classe"].nunique() < 2:
        raise ValueError("É preciso ter exemplos de pelo menos 2 classes diferentes.")

    hog, hog_esp = calcular_hog(dados)
    y_todos = dados["classe"].values
    tr, te, aviso = separar(dados, videos_teste)
    grupos_tr = dados["video"].values[tr]

    candidatos = list(NOMES_MODELOS) if tipo == "auto" else [tipo]
    comparacao = []
    for t in candidatos:
        X = montar_X(dados, hog, USA_HOG[t])
        nota_cv = validacao_cruzada(t, X[tr], y_todos[tr], hog_esp[tr], grupos_tr) if tipo == "auto" else float("nan")
        m = ajustar(t, X[tr], y_todos[tr], hog_esp[tr])
        prev = m.predict(X[te])
        comparacao.append({"tipo": t, "nome": NOMES_MODELOS[t], "cv": nota_cv,
                           "acuracia": accuracy_score(y_todos[te], prev),
                           "acuracia_bal": balanced_accuracy_score(y_todos[te], prev),
                           "modelo": m, "prev": prev})

    # Escolha: no automático, pela validação cruzada nos vídeos de TREINO
    # (escolher pelo teste seria "olhar a prova antes"). Empate: o mais simples.
    if tipo == "auto":
        validos = [c for c in comparacao if not np.isnan(c["cv"])]
        escolhido = max(validos, key=lambda c: c["cv"]) if validos else comparacao[0]
    else:
        escolhido = comparacao[0]
    t = escolhido["tipo"]
    usa_hog = USA_HOG[t]

    y_te = y_todos[te]
    prev = escolhido["prev"]
    X_te_base = dados[car.NOMES_ATRIBUTOS].values[te]
    prev_regra = [car.regra_manual(dict(zip(car.NOMES_ATRIBUTOS, x))) for x in X_te_base]
    classes = [c for c in car.CLASSES.values() if c in set(y_todos)]

    res = {
        "tipo": t, "nome": NOMES_MODELOS[t], "automatico": tipo == "auto", "usa_hog": usa_hog,
        "n_total": len(dados), "n_treino": len(tr), "n_teste": len(te),
        "videos_treino": sorted(set(dados["video"].values[tr])),
        "videos_teste": sorted(set(dados["video"].values[te])),
        "por_classe": dados["classe"].value_counts().to_dict(),
        "origem": exemplos.contar_origem(dados["arquivo"], pasta_dados),
        "acuracia": escolhido["acuracia"], "acuracia_bal": escolhido["acuracia_bal"],
        "acuracia_regra": accuracy_score(y_te, prev_regra),
        "comparacao": [{k: v for k, v in c.items() if k not in ("modelo", "prev")} for c in comparacao],
        "classes": classes,
        "matriz": confusion_matrix(y_te, prev, labels=classes),
        "relatorio": classification_report(y_te, prev, labels=classes, zero_division=0),
        "aviso": aviso,
        "regras": export_text(escolhido["modelo"], feature_names=car.NOMES_ATRIBUTOS) if t == "arvore" else "",
    }
    if t in ("arvore", "floresta"):
        nomes = car.NOMES_ATRIBUTOS + (car.NOMES_HOG if usa_hog else [])
        imp = dict(zip(nomes, escolhido["modelo"].feature_importances_))
        resumo = {n: imp[n] for n in car.NOMES_ATRIBUTOS}
        if usa_hog:
            resumo["forma (HOG)"] = sum(imp[n] for n in car.NOMES_HOG)
        res["importancia"] = sorted(resumo.items(), key=lambda kv: -kv[1])

    # Modelo final: depois de avaliado, é treinado de novo com TODOS os exemplos,
    # para o app usar o máximo de informação disponível.
    X_todos = montar_X(dados, hog, usa_hog)
    final = ajustar(t, X_todos, y_todos, hog_esp)

    os.makedirs(pasta_modelo, exist_ok=True)
    joblib.dump({
        "modelo": final,
        "atributos": car.NOMES_ATRIBUTOS,
        "usa_hog": usa_hog,
        "tipo": t,
        "classes": list(final.classes_),
        "acuracia_teste": res["acuracia"],
        "n_exemplos": len(dados),
        "data": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
    }, os.path.join(pasta_modelo, "modelo_grupo.joblib"))

    _salvar_relatorio(res, pasta_modelo)
    res["figuras"] = _salvar_figuras(res, escolhido["modelo"] if t == "arvore" else None, pasta_modelo)
    return res


def _pct(v):
    return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


def _salvar_relatorio(res, pasta):
    linhas = [
        "DetectAI - Relatório do treinamento",
        f"Data: {datetime.datetime.now():%d/%m/%Y %H:%M}",
        f"Modelo escolhido: {res['nome']}" + (" (escolha automática)" if res["automatico"] else ""),
        "Características: 10 do grupo" + (f" + {car.N_HOG} de forma (HOG)" if res["usa_hog"] else ""),
        "",
        f"Exemplos rotulados: {res['n_total']}  (treino {res['n_treino']}, teste {res['n_teste']})",
        f"Por classe: {res['por_classe']}",
        "Origem dos rótulos: " + ", ".join(f"{NOMES_ORIGENS.get(k, k)}: {v}" for k, v in res["origem"].items()),
        f"Vídeos de treino: {', '.join(map(str, res['videos_treino']))}",
        f"Vídeos de teste:  {', '.join(map(str, res['videos_teste']))}",
    ]
    if res["aviso"]:
        linhas += ["", "ATENÇÃO: " + res["aviso"]]
    linhas += ["", "Comparação dos modelos:",
               f"  {'Modelo':<34}{'Validação (treino)':>20}{'Acurácia (teste)':>18}{'Balanceada (teste)':>20}"]
    for c in res["comparacao"]:
        marca = "  <- escolhido" if c["tipo"] == res["tipo"] else ""
        linhas.append(f"  {c['nome']:<34}{_pct(c['cv']):>20}{_pct(c['acuracia']):>18}{_pct(c['acuracia_bal']):>20}{marca}")
    linhas += [
        f"  {'Regras manuais antigas':<34}{'':>20}{_pct(res['acuracia_regra']):>18}",
        "",
        "Matriz de confusão do modelo escolhido (linha = classe real, coluna = classe prevista):",
        "          " + "".join(f"{c[:8]:>10}" for c in res["classes"]),
    ]
    for c, lin in zip(res["classes"], res["matriz"]):
        linhas.append(f"{c[:8]:>10}" + "".join(f"{v:>10}" for v in lin))
    linhas += ["", "Precisão, revocação e F1 por classe:", res["relatorio"]]
    if res.get("importancia"):
        linhas += ["Importância das características:"]
        linhas += [f"  {n:<14} {v:.3f}" for n, v in res["importancia"]]
    if res["regras"]:
        linhas += ["", "Regras aprendidas pela árvore:", res["regras"]]
    with open(os.path.join(pasta, "relatorio_treino.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def _salvar_figuras(res, arvore, pasta):
    """Gera as figuras para o relatório (se o matplotlib estiver instalado)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.tree import plot_tree
    except ImportError:
        return []
    for antiga in ("arvore.png", "importancia.png"):
        p = os.path.join(pasta, antiga)
        if os.path.exists(p):
            os.remove(p)                     # não deixa figura de um treino anterior
    figuras = []

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=130)
    ConfusionMatrixDisplay(res["matriz"], display_labels=res["classes"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{res['nome']}: acurácia no teste {res['acuracia']:.0%}", fontsize=10)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    fig.tight_layout()
    caminho = os.path.join(pasta, "matriz_confusao.png")
    fig.savefig(caminho)
    plt.close(fig)
    figuras.append(caminho)

    if res.get("importancia"):
        nomes, vals = zip(*res["importancia"][::-1])
        fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
        ax.barh(nomes, vals, color="#2f9e44")
        ax.set_title("Importância de cada característica")
        ax.set_xlabel("Importância (soma = 1)")
        fig.tight_layout()
        caminho = os.path.join(pasta, "importancia.png")
        fig.savefig(caminho)
        plt.close(fig)
        figuras.append(caminho)

    if arvore is not None:
        fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
        plot_tree(arvore, feature_names=car.NOMES_ATRIBUTOS, class_names=list(arvore.classes_),
                  filled=True, rounded=True, max_depth=3, fontsize=8, impurity=False, ax=ax)
        ax.set_title("Árvore de decisão aprendida (3 primeiros níveis)")
        fig.tight_layout()
        caminho = os.path.join(pasta, "arvore.png")
        fig.savefig(caminho)
        plt.close(fig)
        figuras.append(caminho)
    return figuras


# ----------------------------------------------------------------------
# Execução pelo terminal
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treina o modelo do grupo do DetectAI.")
    ap.add_argument("--modelo", choices=["auto"] + list(NOMES_MODELOS), default="auto")
    ap.add_argument("--teste", nargs="*", help="nomes dos vídeos reservados para teste")
    a = ap.parse_args()
    treinar(a.modelo, a.teste)
    print(open(os.path.join(cfg.PASTA_MODELO, "relatorio_treino.txt"), encoding="utf-8").read())
    print(f"\nModelo salvo em {cfg.ARQ_MODELO}")
