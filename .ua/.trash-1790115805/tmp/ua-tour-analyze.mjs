import { readFileSync, writeFileSync } from 'node:fs';

const CODE_ENTRY_FILENAMES = new Set([
  'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js', 'server.ts', 'server.js',
  'mod.rs', 'main.go', 'main.py', 'main.rs', 'manage.py', 'app.py', 'wsgi.py', 'asgi.py',
  'run.py', '__main__.py', 'application.java', 'main.java', 'program.cs', 'config.ru',
  'index.php', 'app.swift', 'application.kt', 'main.cpp', 'main.c'
]);

const NON_CODE_CATEGORIES = {
  documentation: ['document'],
  infrastructure: ['service', 'pipeline', 'resource'],
  data: ['table', 'schema', 'endpoint'],
  config: ['config']
};

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function compareIds(a, b) {
  if (a.id < b.id) return -1;
  if (a.id > b.id) return 1;
  return 0;
}

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  fail('Usage: node ua-tour-analyze.mjs <input.json> <output.json>');
}

let parsed;
try {
  parsed = JSON.parse(readFileSync(inputPath, 'utf8'));
} catch (error) {
  fail(`Failed to parse input JSON at ${inputPath}: ${error.message}`);
}

const nodes = Array.isArray(parsed.nodes) ? parsed.nodes : [];
const edges = Array.isArray(parsed.edges) ? parsed.edges : [];
const layers = Array.isArray(parsed.layers) ? parsed.layers : [];
const nodeById = new Map(nodes.map((node) => [node.id, node]));

const validEdges = edges.filter((edge) => nodeById.has(edge.source) && nodeById.has(edge.target));

const fanIn = new Map(nodes.map((node) => [node.id, 0]));
const fanOut = new Map(nodes.map((node) => [node.id, 0]));
const adjacency = new Map(nodes.map((node) => [node.id, []]));

for (const edge of validEdges) {
  fanIn.set(edge.target, fanIn.get(edge.target) + 1);
  fanOut.set(edge.source, fanOut.get(edge.source) + 1);
  if (edge.type === 'imports' || edge.type === 'calls') {
    adjacency.get(edge.source).push(edge.target);
  }
}

const fanInRanking = nodes
  .map((node) => ({ id: node.id, name: node.name, fanIn: fanIn.get(node.id), summary: node.summary }))
  .sort((a, b) => (b.fanIn - a.fanIn) || compareIds(a, b))
  .slice(0, 20);

const fanOutRanking = nodes
  .map((node) => ({ id: node.id, name: node.name, fanOut: fanOut.get(node.id), summary: node.summary }))
  .sort((a, b) => (b.fanOut - a.fanOut) || compareIds(a, b))
  .slice(0, 20);

const fanInValuesAsc = [...fanIn.values()].sort((a, b) => a - b);
const fanOutValuesDesc = [...fanOut.values()].sort((a, b) => b - a);
const lowFanInThreshold = nodes.length > 0 ? fanInValuesAsc[Math.max(0, Math.ceil(nodes.length * 0.25) - 1)] : 0;
const highFanOutThreshold = nodes.length > 0 ? fanOutValuesDesc[Math.max(0, Math.ceil(nodes.length * 0.1) - 1)] : 0;

const entryPointCandidates = [];
for (const node of nodes) {
  const filePath = node.filePath || node.name || '';
  const depth = filePath.split('/').filter(Boolean).length;
  const baseName = (node.name || '').toLowerCase();
  let score = 0;

  if (node.type === 'file') {
    if (CODE_ENTRY_FILENAMES.has(baseName)) score += 3;
    if (depth <= 2) score += 1;
    if (highFanOutThreshold > 0 && fanOut.get(node.id) >= highFanOutThreshold) score += 1;
    if (fanIn.get(node.id) <= lowFanInThreshold) score += 1;
  } else if (node.type === 'document') {
    if (baseName === 'readme.md' && depth <= 1) score += 5;
    else if (baseName.endsWith('.md') && depth <= 1) score += 2;
  }

  if (score > 0) {
    entryPointCandidates.push({
      id: node.id,
      name: node.name,
      type: node.type,
      score,
      fanIn: fanIn.get(node.id),
      fanOut: fanOut.get(node.id),
      summary: node.summary
    });
  }
}

entryPointCandidates.sort((a, b) => (b.score - a.score) || (b.fanOut - a.fanOut) || compareIds(a, b));
const topEntryCandidates = entryPointCandidates.slice(0, 5);
const codeEntry = entryPointCandidates.find((candidate) => candidate.type === 'file') || null;

const bfsOrder = [];
const depthMap = {};
let bfsStart = null;
if (codeEntry) {
  bfsStart = codeEntry.id;
  const visited = new Set([bfsStart]);
  const queue = [{ id: bfsStart, depth: 0 }];
  while (queue.length > 0) {
    const current = queue.shift();
    bfsOrder.push(current.id);
    depthMap[current.id] = current.depth;
    const neighbors = [...new Set(adjacency.get(current.id) || [])].sort();
    for (const next of neighbors) {
      if (!visited.has(next)) {
        visited.add(next);
        queue.push({ id: next, depth: current.depth + 1 });
      }
    }
  }
}

const byDepth = {};
for (const [id, depth] of Object.entries(depthMap)) {
  const key = String(depth);
  if (!byDepth[key]) byDepth[key] = [];
  byDepth[key].push(id);
}
for (const key of Object.keys(byDepth)) byDepth[key].sort();

const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
for (const node of nodes) {
  for (const [category, types] of Object.entries(NON_CODE_CATEGORIES)) {
    if (types.includes(node.type)) {
      nonCodeFiles[category].push({ id: node.id, name: node.name, type: node.type, summary: node.summary });
      break;
    }
  }
}
for (const key of Object.keys(nonCodeFiles)) {
  nonCodeFiles[key].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

function internalEdgeCount(memberSet) {
  return validEdges.filter((edge) => memberSet.has(edge.source) && memberSet.has(edge.target)).length;
}

const directedPairs = new Set(validEdges.map((edge) => `${edge.source}|${edge.target}`));
const bidirectionalPairs = new Set();
for (const pair of directedPairs) {
  const separatorIndex = pair.indexOf('|');
  const source = pair.slice(0, separatorIndex);
  const target = pair.slice(separatorIndex + 1);
  if (directedPairs.has(`${target}|${source}`)) {
    bidirectionalPairs.add(source < target ? `${source}|${target}` : `${target}|${source}`);
  }
}

const parent = new Map(nodes.map((node) => [node.id, node.id]));
function find(id) {
  let root = id;
  while (parent.get(root) !== root) root = parent.get(root);
  while (parent.get(id) !== root) {
    const next = parent.get(id);
    parent.set(id, root);
    id = next;
  }
  return root;
}
function union(a, b) {
  const rootA = find(a);
  const rootB = find(b);
  if (rootA !== rootB) parent.set(rootA, rootB);
}

for (const pair of bidirectionalPairs) {
  const [a, b] = pair.split('|');
  union(a, b);
}

const components = new Map();
for (const pair of bidirectionalPairs) {
  const [a] = pair.split('|');
  const root = find(a);
  if (!components.has(root)) components.set(root, new Set());
  components.get(root).add(a);
  components.get(root).add(pair.slice(pair.indexOf('|') + 1));
}

function connectionCount(candidateId, memberSet) {
  let count = 0;
  for (const edge of validEdges) {
    if (edge.source === candidateId && memberSet.has(edge.target)) count += 1;
    else if (edge.target === candidateId && memberSet.has(edge.source)) count += 1;
  }
  return count;
}

const clusters = [];
for (const memberSet of components.values()) {
  let changed = true;
  while (changed) {
    changed = false;
    for (const node of nodes) {
      if (memberSet.has(node.id)) continue;
      if (connectionCount(node.id, memberSet) >= 2) {
        memberSet.add(node.id);
        changed = true;
      }
    }
  }

  if (memberSet.size > 5) {
    const ranked = [...memberSet].sort((a, b) => {
      const connectivityA = connectionCount(a, memberSet);
      const connectivityB = connectionCount(b, memberSet);
      if (connectivityB !== connectivityA) return connectivityB - connectivityA;
      return a < b ? -1 : a > b ? 1 : 0;
    });
    memberSet.clear();
    for (const id of ranked.slice(0, 5)) memberSet.add(id);
  }

  if (memberSet.size >= 2) {
    clusters.push({ nodes: [...memberSet].sort(), edgeCount: internalEdgeCount(memberSet) });
  }
}

clusters.sort((a, b) => (b.edgeCount - a.edgeCount) || (a.nodes[0] < b.nodes[0] ? -1 : a.nodes[0] > b.nodes[0] ? 1 : 0));

const nodeSummaryIndex = {};
for (const node of nodes) {
  nodeSummaryIndex[node.id] = { name: node.name, type: node.type, summary: node.summary };
}

const results = {
  scriptCompleted: true,
  entryPointCandidates: topEntryCandidates,
  fanInRanking,
  fanOutRanking,
  bfsTraversal: { startNode: bfsStart, order: bfsOrder, depthMap, byDepth },
  nonCodeFiles,
  clusters: clusters.slice(0, 10),
  layers: { count: layers.length, list: layers },
  nodeSummaryIndex,
  totalNodes: nodes.length,
  totalEdges: edges.length
};

try {
  writeFileSync(outputPath, `${JSON.stringify(results, null, 2)}\n`, 'utf8');
} catch (error) {
  fail(`Failed to write results JSON to ${outputPath}: ${error.message}`);
}

process.exit(0);
