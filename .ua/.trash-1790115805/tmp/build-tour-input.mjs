import { readFileSync, writeFileSync } from 'node:fs';

const [, , graphPath, layersPath, outPath] = process.argv;
const graph = JSON.parse(readFileSync(graphPath, 'utf8'));
const layersRaw = JSON.parse(readFileSync(layersPath, 'utf8'));

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

const nodes = graph.nodes
  .filter((n) => fileLevelTypes.has(n.type))
  .map((n) => ({
    id: n.id,
    type: n.type,
    name: n.name,
    filePath: n.filePath,
    summary: n.summary,
  }));

const nodeIds = new Set(nodes.map((n) => n.id));
const edges = graph.edges
  .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
  .map((e) => ({ source: e.source, target: e.target, type: e.type }));

const layers = (Array.isArray(layersRaw) ? layersRaw : layersRaw.layers || []).map(
  (l) => ({ id: l.id, name: l.name, description: l.description }),
);

const input = { nodes, edges, layers };
writeFileSync(outPath, JSON.stringify(input, null, 2));
console.log(`nodes=${nodes.length} edges=${edges.length} layers=${layers.length}`);
