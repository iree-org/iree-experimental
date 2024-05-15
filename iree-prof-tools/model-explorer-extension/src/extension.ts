import * as vscode from 'vscode';
import {convertMlirToJsonIfNecessary} from './mlirUtil';
import {WebviewPanelForModelExplorer} from './modelExplorer';

export function activate(context: vscode.ExtensionContext) {
  const modelToPanelMap = new Map<string, WebviewPanelForModelExplorer>();

  // Command to start a model explorer associated to the active text editor.
  context.subscriptions.push(
    vscode.commands.registerCommand('modelExplorer.start', async () => {
      const modelFile = await getModelFileName();
      if (!modelFile) {
        vscode.window.showErrorMessage('Invalid model file path.');
        return;
      }

      const panel = new WebviewPanelForModelExplorer(
        context,
        vscode.window.activeTextEditor
      );
      modelToPanelMap.set(modelFile, panel);
      panel.addDisposeCallback(() => { modelToPanelMap.delete(modelFile); });
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

  // Command to focus on a node in the model explorer associated to the active
  // text editor. The model explorer must be launched before with start command
  // above.
  context.subscriptions.push(
    vscode.commands.registerCommand('modelExplorer.focus', () => {
      const modelFile = vscode.window.activeTextEditor?.document.fileName;
      if (!modelFile) {
        return;
      }

      const panel = modelToPanelMap.get(modelFile);
      if (panel) {
        panel.focusOnNode();
      } else {
        vscode.window.showErrorMessage(
          'Model explorer is not running for ' + modelFile
        );
      }
    })
  );
}

// Gets the filename of active text editor. If no editors are active, show the
// file open dialog.
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
