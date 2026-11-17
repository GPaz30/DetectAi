"""
DetectAI - Desenho das imagens exibidas
=======================================
Funções que só desenham por cima das imagens; nenhuma delas muda a detecção.
"""
import cv2
import numpy as np

import config as cfg


def desenhar_objetos(frame, objetos):
    """Etapa 10: desenha, para cada objeto, a caixa com cantos reforçados e o nome."""
    saida = frame.copy()  # copia para não alterar o quadro original
    for o in objetos:
        x, y, w, h = o["caixa"]
        cor = cfg.CORES[o["grupo"]]

        # Retângulo fino ao redor do objeto.
        cv2.rectangle(saida, (x, y), (x + w, y + h), cor, 1, cv2.LINE_AA)

        # Cantos reforçados: duas linhas grossas em cada um dos 4 cantos.
        t = max(8, min(w, h) // 4)   # comprimento de cada linha do canto
        cantos = [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]
        for (px, py, dx, dy) in cantos:
            cv2.line(saida, (px, py), (px + dx * t, py), cor, 3, cv2.LINE_AA)
            cv2.line(saida, (px, py), (px, py + dy * t), cor, 3, cv2.LINE_AA)

        # Etiqueta com o nome: um retângulo colorido com o texto preto por cima.
        texto = o["nome"]
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # A etiqueta fica acima da caixa; se não couber, fica dentro.
        y0 = y - th - 10 if y - th - 10 > 0 else y + 2
        cv2.rectangle(saida, (x, y0), (x + tw + 10, y0 + th + 8), cor, -1)
        cv2.putText(saida, texto, (x + 5, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return saida


def desenhar_contornos(limpa, caixas, area_min, cortes=None):
    """
    Etapa 9: sobre a máscara limpa, mostra
      - em verde, os contornos aceitos (área >= tamanho mínimo);
      - em vermelho, os contornos descartados por serem pequenos;
      - em laranja, as caixas finais;
      - em rosa, as linhas onde objetos grudados foram separados (watershed).
    """
    vis = cv2.cvtColor(limpa // 2, cv2.COLOR_GRAY2BGR)   # máscara escurecida, para destacar as linhas
    contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        cor = (0, 220, 0) if cv2.contourArea(c) >= area_min else (0, 0, 255)
        cv2.drawContours(vis, [c], -1, cor, 1, cv2.LINE_AA)
    for (x, y, w, h) in caixas:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
    if cortes is not None:                       # por cima das caixas, para aparecer
        vis[cv2.dilate(cortes, np.ones((3, 3), np.uint8)) > 0] = (255, 0, 255)
    return vis


def imagem_aviso(texto, largura=320, altura=180):
    """Imagem cinza com um texto, para as etapas que ainda não têm conteúdo."""
    img = np.full((altura, largura, 3), 40, np.uint8)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, texto, ((largura - tw) // 2, (altura + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img
