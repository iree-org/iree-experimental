import * as vscode from 'vscode';
import {convertMlirToJsonIfNecessary} from './mlirUtil';
import {WebviewPanelForModelExplorer} from './modelExplorer';

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand('modelExplorer.show', async () => {
      const modelFile = await getModelFileName();
      if (!modelFile) {
        vscode.window.showInformationMessage('Invalid model file path.');
        return;
      }

      const panel = new WebviewPanelForModelExplorer(context);
      context.subscriptions.push(panel);

      const modelFileToLoad = await convertMlirToJsonIfNecessary(modelFile);
      if (modelFileToLoad != modelFile) {
        panel.addDisposeCallback(() => {
          vscode.workspace.fs.delete(vscode.Uri.file(modelFileToLoad));
        });
      }

      panel.startModelExplorer(modelFileToLoad);
    })
  );
}

async function getModelFileName(): Promise<string | undefined> {
  const fileName = vscode.window.activeTextEditor?.document.fileName;
  if (fileName) {
    return fileName;
  }

  const openFiles = await vscode.window.showOpenDialog({
    canSelectFiles: true,
    canSelectMany: false,
    title: 'No active text editor or too large file. Open a model file'
  });

  return openFiles?.length == 1 ? openFiles[0].fsPath : undefined;
}
