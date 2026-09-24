import { readFileSync, writeFileSync } from 'node:fs';

const [, , scanPath, outPath] = process.argv;
const scan = JSON.parse(readFileSync(scanPath, 'utf8'));
const input = {
  projectRoot: 'D:/kodlar/arealai_case_study',
  files: scan.files.map((f) => ({
    path: f.path,
    language: f.language,
    fileCategory: f.fileCategory,
  })),
};
writeFileSync(outPath, JSON.stringify(input, null, 2));
console.log(`wrote ${outPath} with ${input.files.length} files`);
