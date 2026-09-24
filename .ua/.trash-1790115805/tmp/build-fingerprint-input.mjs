import { readFileSync, writeFileSync } from 'node:fs';

const [, , scanPath, outPath, commitHash] = process.argv;
const scan = JSON.parse(readFileSync(scanPath, 'utf8'));
const input = {
  projectRoot: 'D:/kodlar/arealai_case_study',
  filePaths: scan.files.map((f) => f.path),
  gitCommitHash: commitHash,
};
writeFileSync(outPath, JSON.stringify(input, null, 2));
console.log(`fingerprint input: ${input.filePaths.length} files`);
