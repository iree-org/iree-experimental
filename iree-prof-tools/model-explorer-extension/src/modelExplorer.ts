import * as vscode from 'vscode';

// Internal model explorer web server shared by multiple webview panels.
var internalModelExplorerTerminal: vscode.Terminal | undefined = undefined;

// A random port number of internal model explorer web server.
var internalModelExplorerPort: number | undefined = undefined;

export class WebviewPanelForModelExplorer {
  context: vscode.ExtensionContext;
  panel: vscode.WebviewPanel;
  disposeCallbacks: (() => void)[];

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.panel = vscode.window.createWebviewPanel(
      'modelExplorer',
      'Model Explorer',
      vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.One,
      { // Webview options.
        enableScripts: true,
        retainContextWhenHidden: true
      }
    );

    this.disposeCallbacks = [];

    this.panel.onDidDispose(
      () => { for (let f of this.disposeCallbacks) { f(); }},
      null,
      context.subscriptions);
  }

  dispose() {
    this.panel.dispose();
  }

  addDisposeCallback(f: () => void) {
    this.disposeCallbacks.push(f);
  }

  startModelExplorer(modelFile: string) {
    const config = vscode.workspace.getConfiguration('modelExplorer');
    const externalUrl = config.get<string>('externalModelExplorerUrl') ?? '';
    if (externalUrl.length > 0) {
      this.panel.webview.html = getWebviewContent(externalUrl, modelFile);
      return;
    }

    internalModelExplorerPort = internalModelExplorerPort ?? getRandomPort();
    const modelExplorerUrl = `http://localhost:${internalModelExplorerPort}`;
    if (internalModelExplorerTerminal != null) {
      this.panel.webview.html = getWebviewContent(modelExplorerUrl, modelFile);
      return;
    }

    // No model explorer is available. Starts one.
    vscode.window.showInformationMessage(
      'Starting a model explorer web server...'
    );
    internalModelExplorerTerminal = vscode.window.createTerminal(
      'modelExplorerWebServer',
      config.get<string>('internalModelExplorerPath') ?? 'model-explorer',
      ['--no_open_in_browser', `--port=${internalModelExplorerPort}`]
    );
    const token = vscode.window.onDidCloseTerminal(terminal => {
      if (terminal == internalModelExplorerTerminal) {
        token.dispose();
        vscode.window.showInformationMessage(
          'Model explorer web server is closed.'
        );
        internalModelExplorerTerminal = undefined;
        internalModelExplorerPort = undefined;
      }
    });
    this.context.subscriptions.push(internalModelExplorerTerminal);

    // Delay webview rendering to wait for model explorer ready.
    const timeout = setTimeout(() => {
      this.panel.webview.html = getWebviewContent(modelExplorerUrl, modelFile);
    }, 2000); // 2000 is arbitrary. Need a more reliable way.

    this.addDisposeCallback(() => { clearTimeout(timeout); });
  }
}

function getRandomPort(): number {
  return 30080 + Math.floor(Math.random() * 9900);
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
