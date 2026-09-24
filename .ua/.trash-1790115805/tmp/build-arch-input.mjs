import { readFileSync, writeFileSync } from 'node:fs';

const [, , graphPath, outPath] = process.argv;
const graph = JSON.parse(readFileSync(graphPath, 'utf8'));

const fileLevelTypes = new Set([
  'file',
  'config',
  'document',
  'service',
  'pipeline',
  'table',
  'schema',
  'resource',
  'endpoint',
]);

const fileNodes = graph.nodes
  .filter((n) => fileLevelTypes.has(n.type))
  .map((n) => ({
    id: n.id,
    type: n.type,
    name: n.name,
    filePath: n.filePath,
    summary: n.summary,
    tags: n.tags,
  }));

const fileNodeIds = new Set(fileNodes.map((n) => n.id));
const fileLevelEdges = graph.edges.filter(
  (e) => fileNodeIds.has(e.source) && fileNodeIds.has(e.target),
);

const importEdges = fileLevelEdges
  .filter((e) => e.type === 'imports')
  .map((e) => ({ source: e.source, target: e.target, type: e.type }));

const input = {
  fileNodes,
  importEdges,
  allEdges: fileLevelEdges.map((e) => ({
    source: e.source,
    target: e.target,
    type: e.type,
  })),
};

writeFileSync(outPath, JSON.stringify(input, null, 2));
console.log(
  `fileNodes=${fileNodes.length} importEdges=${importEdges.length} allFileLevelEdges=${input.allEdges.length}`,
);
