import * as vscode from 'vscode';
import {NodeMap, buildNodeMap} from './graphUtil';

// Internal model explorer web server shared by multiple webview panels.
var internalModelExplorerTerminal: vscode.Terminal | undefined = undefined;

// A random port number of internal model explorer web server.
var internalModelExplorerPort: number | undefined = undefined;

// Webview panel to run a model explorer.
export class WebviewPanelForModelExplorer {
  context: vscode.ExtensionContext;
  editor: vscode.TextEditor | undefined;
  panel: vscode.WebviewPanel;
  disposeCallbacks: (() => void)[];
  nodeMap: NodeMap | undefined = undefined;

  constructor(
    context: vscode.ExtensionContext,
    editor: vscode.TextEditor | undefined
  ) {
    this.context = context;
    this.editor = editor;
    this.panel = vscode.window.createWebviewPanel(
      'modelExplorer',
      'Model Explorer',
      vscode.ViewColumn.Beside,
      { // Webview options.
        enableScripts: true,
        retainContextWhenHidden: true
      }
    );

    this.disposeCallbacks = [];

    this.panel.onDidDispose(
      () => { for (let f of this.disposeCallbacks) { f(); } },
      null,
      context.subscriptions
    );
  }

  dispose() {
    this.panel.dispose();
  }

  // Adds a callback called when this webview panel is disposed.
  addDisposeCallback(f: () => void) {
    this.disposeCallbacks.push(f);
  }

  // Starts a model explorer within this webview panel.
  async startModelExplorer(modelFile: string) {
    // Set up a message channel to model explorer in webview for interaction
    // with the editor.
    this.nodeMap = await buildNodeMap(modelFile);
    if (this.nodeMap) {
      this.panel.webview.onDidReceiveMessage(
        async message => { this.onMessage(message); },
        undefined,
        this.context.subscriptions
      );
    }

    // If model explorer is an external server, no need to wait for it ready.
    const config = vscode.workspace.getConfiguration('modelExplorer');
    const externalUrl = config.get<string>('externalModelExplorerUrl') ?? '';
    if (externalUrl.length > 0) {
      this.panel.webview.html = getWebviewContent(externalUrl, modelFile);
      return;
    }

    // If an internal model explorer has already been running, reuse it.
    internalModelExplorerPort = internalModelExplorerPort ?? getRandomPort();
    const modelExplorerUrl = `http://localhost:${internalModelExplorerPort}`;
    if (internalModelExplorerTerminal != null) {
      this.panel.webview.html = getWebviewContent(modelExplorerUrl, modelFile);
      return;
    }

    // No model explorer is available. Start one.
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

  // Called on messages from web explorer in the webview.
  private async onMessage(message: any) {
    console.debug(
      'Got message from model-explorer: ' + JSON.stringify(message, null, 2)
    );
    if (message.cmd == 'model-explorer-node-selected') {
      const node = this.nodeMap?.getNodeById(message.nodeId);
      if (node) {
        this.focusOnText(node.label);
      } else {
        console.error('Unknown node ID: ' + message.nodeId);
      }
    }
  }

  // Sets focus on text matched with the node label of the current node of mode
  // explorer in the webview.
  private async focusOnText(nodeLabel: string | undefined) {
    if (!nodeLabel || !this.editor) {
      return;
    }

    // Find all matches with node label.
    // TODO(byungchul): Utilize source file location info if exists.
    const document = this.editor.document;
    const selections: vscode.Selection[] = [];
    for (const match of document.getText().matchAll(RegExp(nodeLabel, 'g'))) {
      const start = document.positionAt(match.index);
      const end = document.positionAt(match.index + match[0].length);
      selections.push(new vscode.Selection(start, end));
    }

    if (selections.length > 0) {
      // Bring the focus on this editor.
      this.editor = await vscode.window.showTextDocument(
        this.editor.document,
        this.editor.viewColumn
      );
      this.editor.selections = selections;
      this.editor.revealRange(selections[0], vscode.TextEditorRevealType.AtTop);
    }
  }

  // Sets focus on a node of model explorer in the webview.
  focusOnNode() {
    if (!this.editor || !this.nodeMap) {
      return;
    }

    // Find a word at the current cursor position.
    const wordRange = this.editor.document.getWordRangeAtPosition(
      this.editor.selection.active
    );

    const word = wordRange ? this.editor.document.getText(wordRange) : '';
    console.debug('Word at cursor = "' + word + '"');
    if (!word) {
      return;
    }

    const node = this.nodeMap.getFirstNodeByLabel(word);
    if (node) {
      this.panel.webview.postMessage({
        'cmd': 'model-explorer-select-node-by-node-id',
        'nodeId': node.id
      });
      // Bring the focus on the webview panel.
      this.panel.reveal();
    }
  }
}

// Gets a random port for internal model explorer server.
function getRandomPort(): number {
  return 30080 + Math.floor(Math.random() * 9900);
}

// Gets the webview contents, i.e. HTML.
// It wraps model-explorer with iframe, and listens to message both from this
// extension and from the model explorer.
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
  <iframe id="model-explorer-iframe",
          src="${modelExplorerUrl}/?data=${encodedData}&renderer=webgl&show_open_in_new_tab=0"
          style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;">
  </iframe>
  <script>
    const vscode = acquireVsCodeApi();
    const iframeWindow = document.getElementById('model-explorer-iframe').contentWindow;
    window.addEventListener('message', event => {
      const message = event.data;
      console.log('Got message: ' + JSON.stringify(message, null, 2) + ' from ' + event.origin);
      if (event.origin.startsWith('vscode-webview:')) {
        iframeWindow.postMessage(message, '*');
      } else {
        vscode.postMessage(message);
      }
    });
  </script>
</body>
</html>`;
}
