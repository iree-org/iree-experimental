import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    context.subscriptions.push(
      vscode.commands.registerCommand('model-explorer.show', () => {
        const panel = vscode.window.createWebviewPanel(
          'modelExplorer',
          'Model Explorer',
          vscode.ViewColumn.One, // Editor column to show the new webview panel in.
          { // Webview options.
            enableScripts: true
          }
        );

        panel.webview.html = getWebviewContent();
      })
    );
  }

  function getWebviewContent() {
    return `<!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <title>Model Explorer</title>
  </head>
  <body>
      <iframe src="http://localhost:8080"
              style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;">
      </iframe>
  </body>
  </html>`;
}