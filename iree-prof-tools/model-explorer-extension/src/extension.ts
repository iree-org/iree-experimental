import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  // A random port number of model explorer web server.
  const port = 30080 + Math.floor(Math.random() * 9900);

  // Model explorer web server shared by multiple webview panels.
  var modelExplorerWebServerTerminal: vscode.Terminal|null = null

  context.subscriptions.push(
    vscode.commands.registerCommand('model-explorer.show', () => {
      const activeTextEditor = vscode.window.activeTextEditor;
      if (activeTextEditor == null) {
        console.error("No active text editor.");
        return
      }

      // Assume the file of active text editor is of graph json.
      // TODO(byungchul): Convert MLIR files to graph json on the fly.
      const modelGraphJson = activeTextEditor.document.fileName;

      const panel = vscode.window.createWebviewPanel(
        'modelExplorer',
        'Model Explorer',
        activeTextEditor.viewColumn ?? vscode.ViewColumn.One,
        { // Webview options.
          enableScripts: true
        }
      );

      if (modelExplorerWebServerTerminal != null) {
        panel.webview.html = getWebviewContent(port, modelGraphJson);
        return;
      }

      modelExplorerWebServerTerminal =
        vscode.window.createTerminal(
          'modelExplorerWebServer',
          'model-explorer',
          ['--no_open_in_browser', `--port=${port}`]
        )
      vscode.window.onDidCloseTerminal(terminal => {
        if (terminal == modelExplorerWebServerTerminal) {
          console.log("Model explorer web server is closed.");
          modelExplorerWebServerTerminal = null;
        }
      })
      context.subscriptions.push(modelExplorerWebServerTerminal);

      const timeout = setTimeout(() => {
        panel.webview.html = getWebviewContent(port, modelGraphJson);
      }, 2000); // 2000 is arbitrary. Need a more reliable way.

      panel.onDidDispose(() => { clearTimeout(timeout); }, null, context.subscriptions);
    })
  );
}

function getWebviewContent(port: number, modelGraphJson: string) {
  const encodedData = encodeURIComponent(`{"models":[{"url":"${modelGraphJson}","adapterId":"builtin_json"}]}`);
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Model Explorer</title>
</head>
<body>
  <iframe src="http://localhost:${port}/?data=${encodedData}&renderer=webgl&show_open_in_new_tab=0"
          style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;">
  </iframe>
</body>
</html>`;
}