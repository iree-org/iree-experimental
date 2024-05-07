# Model Explorer in VS Code

Model explorer is a web tool to visualize ML models. This extension is to embed
model explorer within VS Code.

## How to run it

Model explorer must be installed before this extension get started.

```
pip install model-explorer
```

This extension is started by executing `Model Explorer` command from the Command Palette
on a VS code active text editor of a graph json file. The extension loads the graph json
file to the model explorer web server. If the model explorer is not running, it starts one
with a random port number in a terminal.

## How to make changes into it

To make changes of this extension, VS Code, Node.js and typescript compiler are necessary.
Please follow installation guides of
[VS Code](https://code.visualstudio.com/docs/setup/setup-overview) and
[Node.js](https://nodejs.org/en/download).

Then, install typescript compiler under the extension directory.

```
cd <iree-experimental-root>/iree-prof-tools/model-explorer-extension
npm install
code .
```

Once the workspace is open, follow the
[steps to build webview extensions](https://code.visualstudio.com/api/extension-guides/webview).