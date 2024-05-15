import * as vscode from 'vscode';

// Converts an IREE MLIR file to a graph JSON file with iree-vis program.
export async function convertMlirToJsonIfNecessary(
  modelFile: string
): Promise<string> {
  const config = vscode.workspace.getConfiguration('modelExplorer');
  const ireeVisPath = config.get<string>('ireeVisPath') ?? '';
  if (!modelFile.endsWith('.mlir') || ireeVisPath.length == 0) {
    return modelFile;
  }

  vscode.window.showInformationMessage(
    `Generating graph for a model file, ${modelFile}...`
  );
  const graphJsonFile = modelFile + '.graph.json';
  const ireeVisTerminal = vscode.window.createTerminal({
    'name': 'ireeVisRunner',
    'shellPath': ireeVisPath,
    'shellArgs': [
      `--input_iree_file=${modelFile}`,
      `--output_json_file=${graphJsonFile}`
    ],
    'hideFromUser': true
  });

  return new Promise<string>((resolve, reject) => {
    const token = vscode.window.onDidCloseTerminal(terminal => {
      if (terminal == ireeVisTerminal) {
        token.dispose();
        if (terminal.exitStatus?.code == 0) {
          vscode.window.showInformationMessage(
            `Succeeded to generate a graph file, ${graphJsonFile}.`
          );
          resolve(graphJsonFile);
        } else {
          vscode.window.showErrorMessage('Failed to generate a graph file.');
          resolve(modelFile);
        }
      }
    });
  });
}
