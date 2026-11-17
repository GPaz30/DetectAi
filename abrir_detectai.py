"""
Abre o DetectAI numa janela própria (sem usar o navegador).

Como funciona:
  1) inicia o Streamlit em segundo plano, sem abrir o navegador
     e sem janela de terminal (as mensagens vão para detectai_log.txt);
  2) espera o servidor ficar pronto;
  3) abre uma janela de programa (pywebview) mostrando a interface;
  4) quando a janela é fechada, encerra o Streamlit.

Requisito extra (só uma vez, com o ambiente .venv ativo):
    pip install pywebview

Executar sem terminal: dois cliques em DetectAI.vbs
Executar vendo as mensagens: python abrir_detectai.py
"""
import os
import socket
import subprocess
import sys
import time

import webview  # pywebview: cria a janela do programa

PORTA = 8501
PASTA = os.path.dirname(os.path.abspath(__file__))   # pasta onde está o app.py
LOG = os.path.join(PASTA, "detectai_log.txt")         # mensagens do Streamlit


def servidor_pronto():
    """Retorna True se o Streamlit já está respondendo na porta."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", PORTA)) == 0


def mostrar_erro(mensagem):
    """Sem terminal, o erro aparece numa janela."""
    html = (f"<body style='font-family:Segoe UI,sans-serif;padding:24px'>"
            f"<h2 style='color:#c92a2a'>DetectAI não iniciou</h2><p>{mensagem}</p>"
            f"<p>Detalhes no arquivo <b>detectai_log.txt</b>, na pasta do projeto.</p></body>")
    webview.create_window("DetectAI - erro", html=html, width=560, height=300)
    webview.start()


# O Streamlit roda com python.exe (pythonw.exe não tem saída de texto e
# pode travar ao escrever mensagens). No Windows, CREATE_NO_WINDOW evita
# que um terminal apareça.
python = sys.executable.replace("pythonw.exe", "python.exe")
sem_janela = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# 1) Inicia o Streamlit sem abrir o navegador (headless).
log = open(LOG, "w", encoding="utf-8")
processo = subprocess.Popen(
    [python, "-m", "streamlit", "run", "app.py",
     "--server.headless", "true", "--server.port", str(PORTA),
     "--client.toolbarMode", "minimal"],   # sem menu nem Deploy no canto superior direito
    cwd=PASTA, stdout=log, stderr=subprocess.STDOUT, creationflags=sem_janela,
)

# 2) Espera até 60 segundos o servidor ficar pronto.
inicio = time.time()
erro = None
while not servidor_pronto():
    if processo.poll() is not None:
        erro = "O Streamlit fechou ao iniciar."
        break
    if time.time() - inicio > 60:
        erro = "O Streamlit demorou demais para iniciar."
        break
    time.sleep(0.5)

# 3) Abre a janela do programa (ou a janela de erro).
try:
    if erro:
        mostrar_erro(erro)
    else:
        webview.create_window("DetectAI", f"http://localhost:{PORTA}",
                              width=1400, height=900, min_size=(900, 600))
        webview.start()
finally:
    # 4) Janela fechada: encerra o Streamlit.
    if processo.poll() is None:
        processo.terminate()
    log.close()
