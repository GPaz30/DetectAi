"""
DetectAI - Acompanhar os objetos entre os quadros
=================================================
A detecção olha um quadro de cada vez. Este arquivo liga os quadros entre si:

  Rastreador          - dá um número (ID) a cada objeto, mede a velocidade e
                        guarda os votos de classe (o nome não fica trocando);
  EstabilizadorCaixas - deixa as caixas da tela firmes: sem tremer e sem piscar.
"""
import config as cfg
from deteccao import iou


class Rastreador:
    """
    Acompanha os objetos de um quadro para o outro.
    Cada caixa é associada ao rastro (objeto) cujo centro estava mais perto no
    quadro anterior. Com isso o rastreador:
      - dá um número (ID) estável para cada objeto;
      - mede a velocidade (quanto o centro andou);
      - guarda os votos de classe, para o nome não ficar trocando.
    """

    def __init__(self, votos_minimos=cfg.VOTOS_MINIMOS, esquecer_apos=8):
        self.quadro = 0
        self.rastros = {}
        self.prox_id = 1
        self.votos_minimos = votos_minimos
        self.esquecer_apos = esquecer_apos

    def atualizar(self, caixas, largura_quadro, passos=1):
        """
        Associa as caixas deste quadro aos rastros. Devolve a lista de IDs, na mesma ordem.
        'passos' = quantos quadros analisados se passaram desde a chamada anterior
        (mais de 1 quando a reprodução em tempo real pulou quadros): a velocidade
        continua medida por quadro, e não fica maior só porque houve pulo.
        """
        self.quadro += passos
        livres = set(self.rastros)
        ids = []
        for (x, y, w, h) in caixas:
            cx, cy = x + w / 2.0, y + h / 2.0
            melhor, melhor_d = None, None
            for tid in livres:
                tx, ty = self.rastros[tid]["c"]
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d <= max(w, h) * passos + 30 and (melhor_d is None or d < melhor_d):
                    melhor, melhor_d = tid, d
            if melhor is None:                       # objeto novo
                melhor = self.prox_id
                self.prox_id += 1
                self.rastros[melhor] = {"c": (cx, cy), "vel": 0.0, "idade": 0, "votos": {}, "visto": 0}
            else:                                    # objeto já conhecido: atualiza a velocidade
                livres.discard(melhor)
                r = self.rastros[melhor]
                desloc = ((cx - r["c"][0]) ** 2 + (cy - r["c"][1]) ** 2) ** 0.5 / largura_quadro / passos
                r["vel"] = 0.6 * r["vel"] + 0.4 * desloc   # média móvel (suaviza a medida)
            r = self.rastros[melhor]
            r["c"], r["visto"], r["idade"] = (cx, cy), self.quadro, r["idade"] + 1
            ids.append(melhor)

        # Esquece os objetos que sumiram há vários quadros.
        for tid in list(self.rastros):
            if self.quadro - self.rastros[tid]["visto"] > self.esquecer_apos:
                del self.rastros[tid]
        return ids

    def velocidade(self, tid):
        """Velocidade do objeto (0 nos primeiros quadros, quando ainda não é confiável)."""
        r = self.rastros[tid]
        return r["vel"] if r["idade"] >= 3 else 0.0

    def idade(self, tid):
        """Em quantos quadros o objeto já apareceu."""
        return self.rastros[tid]["idade"]

    def votar(self, tid, nome, probabilidades=None):
        """
        Registra o voto deste quadro e devolve o nome mais votado até agora.
        Com 'probabilidades' ({classe: chance}), o voto é ponderado: um quadro em
        que o modelo tinha 90% de certeza pesa mais que um com 40%.
        """
        r = self.rastros[tid]
        # Os votos antigos perdem peso a cada quadro: se o objeto ficou grudado em
        # outro por alguns quadros (pessoa + carro = "veículo"), o nome se corrige
        # logo depois que eles se separam, em vez de carregar aqueles votos para sempre.
        for classe in r["votos"]:
            r["votos"][classe] *= cfg.VOTOS_ESQUECIMENTO
        for classe, peso in (probabilidades or {nome: 1.0}).items():
            r["votos"][classe] = r["votos"].get(classe, 0.0) + peso
        r["n_votos"] = r.get("n_votos", 0) + 1
        if r["n_votos"] >= self.votos_minimos:
            return max(r["votos"], key=r["votos"].get)
        return nome


class EstabilizadorCaixas:
    """
    Deixa as caixas da tela estáveis entre os quadros.
    Sem isso, a caixa "treme" e muda de tamanho a cada quadro, porque a máscara de
    movimento nunca sai exatamente igual, e às vezes pisca (some num quadro e volta).

    Para cada objeto na tela, guarda o centro, a velocidade e o tamanho. A cada quadro:
      1) PREVÊ onde ele deve estar (posição anterior + velocidade);
      2) associa cada caixa medida à previsão mais parecida (maior sobreposição);
      3) CORRIGE a previsão só em parte na direção da medida (filtro alfa-beta):
         a caixa segue o objeto sem tremer e sem ficar para trás;
      4) se o objeto não foi medido, mantém a caixa prevista por alguns quadros;
      5) só mostra um objeto novo depois de vê-lo em CONFIRMACAO quadros seguidos
         (elimina caixas que aparecem um quadro só, por ruído).
    """

    def __init__(self):
        self.trilhas = []

    @staticmethod
    def _caixa(t, prevista=False, passos=1):
        cx, cy = (t["c"][0] + t["v"][0] * passos, t["c"][1] + t["v"][1] * passos) if prevista else t["c"]
        w, h = t["s"]
        return (cx - w / 2, cy - h / 2, w, h)

    @staticmethod
    def _proximidade(caixa, prevista, passos):
        """
        Usada só quando quadros foram pulados (passos > 1): um objeto pequeno pode
        andar mais que a própria largura entre dois quadros analisados, e aí a
        caixa nova nem encosta na anterior. Se os centros estão a menos de meia
        caixa por quadro pulado, as duas são consideradas do mesmo objeto.
        Devolve de 0.1 (no limite) a 0.2 (mesmo centro), abaixo de qualquer
        sobreposição real (>= 0.2); 0 se estiverem longe.
        """
        x, y, w, h = caixa
        px, py, pw, ph = prevista
        d = ((x + w / 2 - px - pw / 2) ** 2 + (y + h / 2 - py - ph / 2) ** 2) ** 0.5
        limite = 0.5 * max(pw, ph) * passos
        return 0.1 + 0.1 * (1 - d / limite) if d <= limite else 0.0

    def atualizar(self, objetos, passos=1):
        """
        Recebe os objetos medidos neste quadro e devolve os objetos a exibir.
        'passos' = quadros analisados desde a chamada anterior (ver Rastreador.atualizar):
        a previsão anda 'passos' vezes a velocidade.
        """
        # 1) e 2) associa cada medida à trilha prevista mais parecida
        pares = []
        for i, o in enumerate(objetos):
            for j, t in enumerate(self.trilhas):
                prevista = self._caixa(t, prevista=True, passos=passos)
                v = iou(o["caixa"], prevista)
                if v < 0.2:
                    # Sem sobreposição suficiente. Só com quadros pulados vale a proximidade.
                    v = self._proximidade(o["caixa"], prevista, passos) if passos > 1 else 0.0
                if v > 0:
                    pares.append((v + (0.1 if o["grupo"] == t["grupo"] else 0), i, j))
        usados_o, usados_t = set(), set()
        for _, i, j in sorted(pares, reverse=True):
            if i in usados_o or j in usados_t:
                continue
            usados_o.add(i)
            usados_t.add(j)
            t, (x, y, w, h) = self.trilhas[j], objetos[i]["caixa"]
            # 3) filtro alfa-beta: corrige a previsão em parte na direção da medida
            px, py = t["c"][0] + t["v"][0] * passos, t["c"][1] + t["v"][1] * passos
            rx, ry = x + w / 2 - px, y + h / 2 - py
            t["c"] = (px + cfg.SUAVE_POSICAO * rx, py + cfg.SUAVE_POSICAO * ry)
            t["v"] = (t["v"][0] + cfg.SUAVE_VELOCIDADE * rx / passos, t["v"][1] + cfg.SUAVE_VELOCIDADE * ry / passos)
            t["s"] = (t["s"][0] + cfg.SUAVE_TAMANHO * (w - t["s"][0]),
                      t["s"][1] + cfg.SUAVE_TAMANHO * (h - t["s"][1]))
            t.update(nome=objetos[i]["nome"], grupo=objetos[i]["grupo"], perdidos=0, vistos=t["vistos"] + 1)

        # 4) trilhas sem medida: seguem a previsão por alguns quadros e depois somem
        for j, t in enumerate(self.trilhas):
            if j not in usados_t:
                t["c"] = (t["c"][0] + t["v"][0] * passos, t["c"][1] + t["v"][1] * passos)
                t["perdidos"] += passos
                if t["vistos"] < cfg.CONFIRMACAO:
                    t["perdidos"] = cfg.PERSISTENCIA + 1     # nem chegou a ser confirmada: descarta já
        self.trilhas = [t for t in self.trilhas if t["perdidos"] <= cfg.PERSISTENCIA]

        # medidas novas viram trilhas novas
        for i, o in enumerate(objetos):
            if i not in usados_o:
                x, y, w, h = o["caixa"]
                self.trilhas.append({"c": (x + w / 2, y + h / 2), "v": (0.0, 0.0), "s": (w, h),
                                     "nome": o["nome"], "grupo": o["grupo"], "vistos": 1, "perdidos": 0})

        # 5) só aparecem na tela as trilhas confirmadas
        saida = []
        for t in self.trilhas:
            if t["vistos"] >= cfg.CONFIRMACAO:
                x, y, w, h = self._caixa(t)
                saida.append({"caixa": (int(round(x)), int(round(y)), max(int(round(w)), 1), max(int(round(h)), 1)),
                              "nome": t["nome"], "grupo": t["grupo"]})
        return saida
