import { readFileSync, writeFileSync } from 'node:fs';

const [, , graphPath, layersPath, tourPath, outPath, commitHash] = process.argv;

const graph = JSON.parse(readFileSync(graphPath, 'utf8'));
const layersRaw = JSON.parse(readFileSync(layersPath, 'utf8'));
const tourRaw = JSON.parse(readFileSync(tourPath, 'utf8'));

const nodeIds = new Set(graph.nodes.map((n) => n.id));

function unwrap(value, key) {
  if (Array.isArray(value)) return value;
  if (value && Array.isArray(value[key])) return value[key];
  return [];
}

const knownPrefixes = [
  'file:',
  'config:',
  'document:',
  'service:',
  'pipeline:',
  'table:',
  'schema:',
  'resource:',
  'endpoint:',
  'function:',
  'class:',
  'module:',
  'concept:',
];

function normalizeId(id) {
  if (typeof id !== 'string' || id.length === 0) return '';
  if (knownPrefixes.some((p) => id.startsWith(p))) return id;
  return `file:${id}`;
}

const layers = unwrap(layersRaw, 'layers').map((layer) => {
  const raw = layer.nodeIds ?? layer.nodes ?? [];
  const ids = raw
    .map((entry) =>
      typeof entry === 'string' ? entry : entry && typeof entry.id === 'string' ? entry.id : '',
    )
    .map(normalizeId)
    .filter((id) => id.length > 0 && nodeIds.has(id));
  return {
    id: typeof layer.id === 'string' && layer.id.length > 0 ? layer.id : `layer:${String(layer.name || 'unnamed').toLowerCase().replace(/[^a-z0-9]+/g, '-')}`,
    name: layer.name,
    description: layer.description,
    nodeIds: [...new Set(ids)],
  };
});

const tour = unwrap(tourRaw, 'steps')
  .map((step) => {
    const ids = (step.nodeIds ?? step.nodesToInspect ?? [])
      .map(normalizeId)
      .filter((id) => id.length > 0 && nodeIds.has(id));
    const out = {
      order: step.order,
      title: step.title,
      description: step.description ?? step.whyItMatters,
      nodeIds: [...new Set(ids)],
    };
    if (typeof step.languageLesson === 'string' && step.languageLesson.length > 0) {
      out.languageLesson = step.languageLesson;
    }
    return out;
  })
  .sort((a, b) => a.order - b.order);

const assigned = new Map();
for (const layer of layers) {
  for (const id of layer.nodeIds) {
    if (assigned.has(id)) {
      console.error(`ERROR: node '${id}' assigned to multiple layers`);
    }
    assigned.set(id, layer.id);
  }
}

const projectDescription =
  'Offline Python document-intelligence pipeline for mortgage loan PDFs: PaddleOCR page text extraction, LLM-based page classification with a confidence verifier, LM Studio VLM field extraction, and adjacent-page matching, orchestrated by pipeline.py with per-stage CLI scripts, workflow docs, and a pytest suite.';

const output = {
  version: '1.0.0',
  project: {
    name: 'arealai_case_study',
    languages: ['config', 'markdown', 'python', 'txt'],
    frameworks: ['PaddleOCR', 'PyMuPDF', 'pytest'],
    description: projectDescription,
    analyzedAt: new Date().toISOString(),
    gitCommitHash: commitHash,
  },
  nodes: graph.nodes,
  edges: graph.edges,
  layers,
  tour,
};

writeFileSync(outPath, JSON.stringify(output, null, 2));
console.log(
  `assembled: nodes=${output.nodes.length} edges=${output.edges.length} layers=${layers.length} tour=${tour.length}`,
);
