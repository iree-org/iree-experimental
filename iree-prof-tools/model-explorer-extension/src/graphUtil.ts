import * as vscode from 'vscode';

// Map either by node id or by node label to nodes.
// Note that node ids are unique while node labels are not, i.e. multiple nodes
// may have the same node labels.
// TODO(byungchul): Check if node IDs are unique within a graph collection or
// within a graph.
export class NodeMap {
  private byId: Map<string, any>;
  private byLabel: Map<string, any[]>;

  constructor(graphCollection: any) {
    this.byId = new Map<string, any>();
    this.byLabel = new Map<string, any[]>();
    for (let graph of graphCollection.graphs) {
      for (let node of graph.nodes) {
        this.byId.set(node.id, node);
        const nodes = this.byLabel.get(node.label);
        if (nodes) {
          nodes.push(node);
        } else {
          this.byLabel.set(node.label, [node]);
        }
      }
    }
  }

  // Gets a node or undefined if not found with node ID.
  getNodeById(nodeId: string): any | undefined {
    return this.byId.get(nodeId);
  }

  // Gets the first node or undefined if not found with node label.
  getFirstNodeByLabel(nodeLabel: string): any | undefined {
    const nodes = this.byLabel.get(nodeLabel);
    return nodes && nodes.length > 0 ? nodes[0] : undefined;
  }
}

// Builds NodeMap from a graph JSON file.
// Returns undefined if it fails to parse graph JSON.
export async function buildNodeMap(
  modelFile: string
): Promise<NodeMap | undefined> {
  if (!modelFile.endsWith('.json')) {
    return undefined;
  }

  const blob = await vscode.workspace.fs.readFile(vscode.Uri.file(modelFile));
  const graphCollection = JSON.parse(blob.toString());
  return graphCollection ? new NodeMap(graphCollection) : undefined;
}
