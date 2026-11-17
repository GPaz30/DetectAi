' Abre o DetectAI numa janela propria, SEM terminal (Windows).
' Dois cliques neste arquivo. Ele precisa ficar na mesma pasta do app.py e da pasta .venv.
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
pasta = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = pasta
pythonw = pasta & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pythonw) Then
  MsgBox "Nao encontrei a pasta .venv ao lado deste arquivo." & vbCrLf & _
         "Crie o ambiente e instale as bibliotecas antes.", vbExclamation, "DetectAI"
  WScript.Quit
End If
' 0 = sem janela de terminal; False = nao espera terminar
shell.Run """" & pythonw & """ """ & pasta & "\abrir_detectai.py""", 0, False
