import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  // A random port number of model explorer web server.
  const port = 30080 + Math.floor(Math.random() * 9900);

  // Internal model explorer web server shared by multiple webview panels.
  var internalModelExplorerTerminal: vscode.Terminal|null = null

  async function startModelExplorer() {
    var modelFile = vscode.window.activeTextEditor?.document.fileName;
    if (!modelFile) {
      const openFiles = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectMany: false,
        title: 'No active text editor or too large file. Open a model file'
      });
      modelFile = openFiles?.length == 1 ? openFiles[0].fsPath : undefined;
      if (!modelFile) {
        vscode.window.showInformationMessage('Invalid model file path.');
        return;
      }
    }

    const config = vscode.workspace.getConfiguration('modelExplorer');
    const externalUrl = config.get<string>('externalModelExplorerUrl') ?? '';
    const connectToExternalServer = externalUrl.length > 0;
    const modelExplorerUrl = connectToExternalServer ? externalUrl : `http://localhost:${port}`;

    const panel = vscode.window.createWebviewPanel(
      'modelExplorer',
      'Model Explorer',
      vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.One,
      { // Webview options.
        enableScripts: true
      }
    );
    if (connectToExternalServer || internalModelExplorerTerminal != null) {
      panel.webview.html = getWebviewContent(modelExplorerUrl, modelFile);
      return;
    }

    // No model explorer is available. Starts one.
    vscode.window.showInformationMessage('Starting a model explorer web server...');
    internalModelExplorerTerminal =
      vscode.window.createTerminal(
        'modelExplorerWebServer',
        config.get<string>('internalModelExplorerPath') ?? 'model-explorer',
        ['--no_open_in_browser', `--port=${port}`]
      );
    vscode.window.onDidCloseTerminal(terminal => {
      if (terminal == internalModelExplorerTerminal) {
        vscode.window.showInformationMessage('Model explorer web server is closed.');
        internalModelExplorerTerminal = null;
      }
    });
    context.subscriptions.push(internalModelExplorerTerminal);

    // Delay webview rendering to wait for model explorer ready.
    const timeout = setTimeout(() => {
      panel.webview.html = getWebviewContent(modelExplorerUrl, modelFile!!);
    }, 2000); // 2000 is arbitrary. Need a more reliable way.

    panel.onDidDispose(() => { clearTimeout(timeout); }, null, context.subscriptions);
  }

  context.subscriptions.push(
    vscode.commands.registerCommand('modelExplorer.show', startModelExplorer));
}

function getWebviewContent(modelExplorerUrl: string, modelFile: string) {
  vscode.window.showInformationMessage(`Loading a model file, ${modelFile}...`);
  const encodedData = encodeURIComponent(`{"models":[{"url":"${modelFile}"}]}`);
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Model Explorer</title>
</head>
<body>
  <iframe src="${modelExplorerUrl}/?data=${encodedData}&renderer=webgl&show_open_in_new_tab=0"
          style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;">
  </iframe>
</body>
</html>`;
}