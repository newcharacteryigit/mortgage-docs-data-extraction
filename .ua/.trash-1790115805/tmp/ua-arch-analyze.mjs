import { readFileSync, writeFileSync } from "node:fs";

const [, , inputPath, outputPath] = process.argv;

if (!inputPath || !outputPath) {
  console.error("usage: node ua-arch-analyze.mjs <input.json> <output.json>");
  process.exit(1);
}

function fail(message) {
  console.error(message);
  process.exit(1);
}

let input;
try {
  input = JSON.parse(readFileSync(inputPath, "utf8"));
} catch (error) {
  fail(`failed to read input JSON: ${error.message}`);
}

const fileNodes = Array.isArray(input.fileNodes) ? input.fileNodes : [];
const importEdges = Array.isArray(input.importEdges) ? input.importEdges : [];
const allEdges = Array.isArray(input.allEdges) ? input.allEdges : [];

const nodeById = new Map(fileNodes.map((n) => [n.id, n]));

function normalizePath(p) {
  return String(p ?? "").replace(/\\/g, "/");
}

function nodeFilePath(node) {
  return normalizePath(node.filePath ?? node.name ?? "");
}

// ---------------------------------------------------------------- A. groups

const paths = fileNodes.map(nodeFilePath);
const splitPaths = paths.map((p) => p.split("/"));

function computeCommonPrefix(segmentsList) {
  if (segmentsList.length === 0) return [];
  const first = segmentsList[0];
  const maxLen = Math.min(...segmentsList.map((s) => s.length)) - 1;
  const prefix = [];
  for (let i = 0; i < maxLen; i += 1) {
    const seg = first[i];
    if (segmentsList.every((s) => s[i] === seg)) {
      prefix.push(seg);
    } else {
      break;
    }
  }
  return prefix;
}

const commonPrefix = computeCommonPrefix(splitPaths);

function groupOf(filePath) {
  const segments = normalizePath(filePath).split("/");
  const rest = segments.slice(commonPrefix.length);
  if (rest.length <= 1) return "root";
  return rest[0];
}

const directoryGroups = {};
const nodeGroupMap = {};
for (const node of fileNodes) {
  const group = groupOf(nodeFilePath(node));
  if (!directoryGroups[group]) directoryGroups[group] = [];
  directoryGroups[group].push(node.id);
  nodeGroupMap[node.id] = group;
}

// ------------------------------------------------------- B. node type groups

const nodeTypeGroups = {};
const nodeTypeMap = {};
for (const node of fileNodes) {
  const type = node.type ?? "file";
  if (!nodeTypeGroups[type]) nodeTypeGroups[type] = [];
  nodeTypeGroups[type].push(node.id);
  nodeTypeMap[node.id] = type;
}

// ---------------------------------------------- C. import adjacency + stats

const fanOut = {};
const fanIn = {};
for (const node of fileNodes) {
  fanOut[node.id] = 0;
  fanIn[node.id] = 0;
}
for (const edge of importEdges) {
  if (edge.source in fanOut) fanOut[edge.source] += 1;
  if (edge.target in fanIn) fanIn[edge.target] += 1;
}

// ------------------------------------------ D. cross-category dependency map

const crossCategoryCounts = new Map();
for (const edge of allEdges) {
  const fromType = nodeTypeMap[edge.source] ?? "unknown";
  const toType = nodeTypeMap[edge.target] ?? "unknown";
  const key = `${fromType}->${toType}->${edge.type}`;
  crossCategoryCounts.set(key, (crossCategoryCounts.get(key) ?? 0) + 1);
}
const crossCategoryEdges = [...crossCategoryCounts.entries()]
  .map(([key, count]) => {
    const [fromType, toType, edgeType] = key.split("->");
    return { fromType, toType, edgeType, count };
  })
  .sort((a, b) => b.count - a.count);

// ------------------------------------- E/F. inter-group imports + densities

const pairCounts = new Map();
for (const edge of importEdges) {
  const from = nodeGroupMap[edge.source];
  const to = nodeGroupMap[edge.target];
  if (!from || !to) continue;
  const key = `${from}->${to}`;
  pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
}
const interGroupImports = [...pairCounts.entries()]
  .map(([key, count]) => {
    const [from, to] = key.split("->");
    return { from, to, count };
  })
  .sort((a, b) => b.count - a.count);

const intraGroupDensity = {};
for (const group of Object.keys(directoryGroups)) {
  let internalEdges = 0;
  let totalEdges = 0;
  for (const edge of importEdges) {
    const from = nodeGroupMap[edge.source];
    const to = nodeGroupMap[edge.target];
    const involves = from === group || to === group;
    if (!involves) continue;
    totalEdges += 1;
    if (from === group && to === group) internalEdges += 1;
  }
  intraGroupDensity[group] = {
    internalEdges,
    totalEdges,
    density: totalEdges === 0 ? 0 : Number((internalEdges / totalEdges).toFixed(4)),
  };
}

// ------------------------------------------------ G. pattern classification

const DIRECTORY_PATTERNS = [
  { label: "api", names: ["routes", "api", "controllers", "endpoints", "handlers", "controller", "routers", "blueprints", "serializers"] },
  { label: "service", names: ["services", "core", "lib", "domain", "logic", "internal", "signals", "composables", "mailers", "jobs", "channels"] },
  { label: "data", names: ["models", "db", "data", "persistence", "repository", "entities", "migrations", "sql", "database", "schema", "entity"] },
  { label: "ui", names: ["components", "views", "pages", "ui", "layouts", "screens"] },
  { label: "middleware", names: ["middleware", "plugins", "interceptors", "guards"] },
  { label: "utility", names: ["utils", "helpers", "common", "shared", "tools", "templatetags", "pkg"] },
  { label: "config", names: ["config", "constants", "env", "settings", "management", "commands"] },
  { label: "test", names: ["__tests__", "test", "tests", "spec", "specs"] },
  { label: "types", names: ["types", "interfaces", "schemas", "contracts", "dtos", "dto", "request", "response"] },
  { label: "hooks", names: ["hooks"] },
  { label: "state", names: ["store", "state", "reducers", "actions", "slices"] },
  { label: "assets", names: ["assets", "static", "public"] },
  { label: "entry", names: ["cmd", "bin", "src/main/java"] },
  { label: "documentation", names: ["docs", "documentation", "wiki"] },
  { label: "infrastructure", names: ["deploy", "deployment", "infra", "infrastructure", "k8s", "kubernetes", "helm", "charts", "terraform", "tf", "docker"] },
  { label: "ci-cd", names: [".github", ".gitlab", ".circleci"] },
];

function matchDirectoryPattern(name) {
  const lower = String(name).toLowerCase();
  for (const entry of DIRECTORY_PATTERNS) {
    if (entry.names.includes(lower)) return entry.label;
  }
  return null;
}

function matchFilePattern(node) {
  const p = nodeFilePath(node);
  const base = p.split("/").pop() ?? "";
  const lower = base.toLowerCase();

  if (/^test_.*\.py$/.test(lower) || /\.test\./.test(lower) || /\.spec\./.test(lower) ||
      /_test\.go$/.test(lower) || /Test\.java$/.test(base) || /_spec\.rb$/.test(lower) ||
      /Test\.php$/.test(base) || /Tests\.cs$/.test(base)) {
    return "test";
  }
  if (/\.d\.ts$/.test(lower)) return "types";
  if (lower === "dockerfile" || lower.startsWith("docker-compose")) return "infrastructure";
  if (/\.tf$/.test(lower) || /\.tfvars$/.test(lower)) return "infrastructure";
  if (lower === "makefile") return "infrastructure";
  if (/\.sql$/.test(lower)) return "data";
  if (/\.(graphql|gql|proto)$/.test(lower)) return "types";
  if (/\.(md|rst)$/.test(lower)) return "documentation";
  if (lower === "requirements.txt" || lower === "pyproject.toml" || lower === "setup.py" ||
      lower === "cargo.toml" || lower === "go.mod" || lower === "gemfile" ||
      lower === "pom.xml" || lower === "build.gradle" || lower === "composer.json" ||
      lower === "package.json") {
    return "config";
  }
  if (base === ".env.example" || lower.startsWith(".env")) return "config";
  if (lower === "manage.py" || lower === "config.ru" || lower === "application.java" ||
      lower === "program.cs" || lower === "main.rs" || lower === "lib.rs" || lower === "main.go") {
    return "entry";
  }
  if (lower === "wsgi.py" || lower === "asgi.py") return "config";
  const nodeType = nodeTypeMap[node.id];
  if (nodeType === "config") return "config";
  if (nodeType === "document") return "documentation";
  if (nodeType === "service" || nodeType === "resource") return "infrastructure";
  if (nodeType === "pipeline") return "ci-cd";
  if (nodeType === "table" || nodeType === "schema" || nodeType === "endpoint") return "data";
  return null;
}

const filePatternMatches = {};
for (const node of fileNodes) {
  const match = matchFilePattern(node) ?? "unknown";
  filePatternMatches[node.id] = match;
}

const patternMatches = {};
for (const group of Object.keys(directoryGroups)) {
  const dirMatch = matchDirectoryPattern(group);
  if (dirMatch) {
    patternMatches[group] = dirMatch;
    continue;
  }
  const counts = new Map();
  for (const id of directoryGroups[group]) {
    const label = filePatternMatches[id];
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  let best = "unknown";
  let bestCount = -1;
  for (const [label, count] of [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    if (count > bestCount) {
      best = label;
      bestCount = count;
    }
  }
  patternMatches[group] = best;
}

// ------------------------------------------- H. deployment topology

const infraFiles = fileNodes
  .filter((n) => {
    const base = (nodeFilePath(n).split("/").pop() ?? "").toLowerCase();
    return base === "dockerfile" || base.startsWith("docker-compose") ||
      /\.(tf|tfvars)$/.test(base) || base === "makefile";
  })
  .map((n) => n.filePath);
const ciFiles = fileNodes
  .filter((n) => {
    const p = nodeFilePath(n);
    return p.startsWith(".github/workflows/") || p.startsWith(".gitlab/") || /jenkinsfile/i.test(p);
  })
  .map((n) => n.filePath);
const deploymentTopology = {
  hasDockerfile: infraFiles.some((f) => (f.split("/").pop() ?? "").toLowerCase() === "dockerfile"),
  hasCompose: infraFiles.some((f) => (f.split("/").pop() ?? "").toLowerCase().startsWith("docker-compose")),
  hasK8s: fileNodes.some((n) => matchFilePattern(n) === "infrastructure" && !infraFiles.includes(n.filePath)),
  hasTerraform: infraFiles.some((f) => /\.tf(vars)?$/.test(f.toLowerCase())),
  hasCI: ciFiles.length > 0,
  infraFiles: [...infraFiles, ...ciFiles],
};

// ------------------------------------------------ I. data pipeline detection

const byPattern = (label) =>
  fileNodes.filter((n) => filePatternMatches[n.id] === label).map((n) => n.filePath);
const dataPipeline = {
  schemaFiles: byPattern("types").filter((f) => /\.(graphql|gql|proto)$/i.test(f)),
  migrationFiles: fileNodes
    .filter((n) => /migration/i.test(nodeFilePath(n)))
    .map((n) => n.filePath),
  dataModelFiles: byPattern("data"),
  apiHandlerFiles: byPattern("api"),
};

// ------------------------------------------------ J. documentation coverage

const groupsWithDocs = [];
const undocumentedGroups = [];
for (const group of Object.keys(directoryGroups)) {
  const hasDoc = directoryGroups[group].some((id) => filePatternMatches[id] === "documentation");
  if (hasDoc) groupsWithDocs.push(group);
  else undocumentedGroups.push(group);
}
const totalGroups = Object.keys(directoryGroups).length;
const docCoverage = {
  groupsWithDocs: groupsWithDocs.length,
  totalGroups,
  coverageRatio: totalGroups === 0 ? 0 : Number((groupsWithDocs.length / totalGroups).toFixed(2)),
  undocumentedGroups,
};

// -------------------------------------------------- K. dependency direction

const dependencyDirection = [];
const groupNames = Object.keys(directoryGroups);
for (let i = 0; i < groupNames.length; i += 1) {
  for (let j = i + 1; j < groupNames.length; j += 1) {
    const a = groupNames[i];
    const b = groupNames[j];
    const aToB = pairCounts.get(`${a}->${b}`) ?? 0;
    const bToA = pairCounts.get(`${b}->${a}`) ?? 0;
    if (aToB > bToA) dependencyDirection.push({ dependent: a, dependsOn: b, edgeCount: aToB });
    else if (bToA > aToB) dependencyDirection.push({ dependent: b, dependsOn: a, edgeCount: bToA });
  }
}

// ---------------------------------------------------------------- outputs

const filesPerGroup = {};
for (const [group, ids] of Object.entries(directoryGroups)) {
  filesPerGroup[group] = ids.length;
}
const nodeTypeCounts = {};
for (const [type, ids] of Object.entries(nodeTypeGroups)) {
  nodeTypeCounts[type] = ids.length;
}

const results = {
  scriptCompleted: true,
  projectRoot: process.cwd(),
  commonPrefix: commonPrefix.join("/"),
  directoryGroups,
  nodeTypeGroups,
  nodeGroupMap,
  nodeTypeMap,
  crossCategoryEdges,
  interGroupImports,
  intraGroupDensity,
  patternMatches,
  filePatternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats: {
    totalFileNodes: fileNodes.length,
    filesPerGroup,
    nodeTypeCounts,
  },
  fileFanIn: fanIn,
  fileFanOut: fanOut,
};

try {
  writeFileSync(outputPath, JSON.stringify(results, null, 2), "utf8");
} catch (error) {
  fail(`failed to write results JSON: ${error.message}`);
}

console.log(`analysis complete: ${fileNodes.length} file nodes, ${Object.keys(directoryGroups).length} directory groups`);
process.exit(0);
