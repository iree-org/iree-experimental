# Model Explorer in VS Code

Model explorer is a web tool to visualize ML models. This extension is to embed
model explorer within VS Code.

## How to run it

Model explorer is running as a web server. Once it's installed on a local computer,
it can be running as a web server and accessed via http://localhost:8080.

```
pip install model-explorer
model-explorer --no_open_in_browser
```

Port number can be changed with `--port <port>` option.

Model explorer VS Code extension connects to the web server.

## How to make changes into it

To make changes of this extension, VS Code, Node.js and typescript compiler are necessary.
Please follow installation guides of
[VS Code](https://code.visualstudio.com/docs/setup/setup-overview) and
[Node.js](https://nodejs.org/en/download).

Then, install typescript compiler under the extension directory.

```
cd <iree-experimental-root>/iree-prof-tools/model-explorer-extension
npm install -D typescript
code .
```

Once the workspace is open, follow the
[steps to build webview extensions](https://code.visualstudio.com/api/extension-guides/webview).
