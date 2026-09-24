import { readFileSync, writeFileSync } from 'node:fs';

const [, , batchesPath, tmpDir] = process.argv;
const batches = JSON.parse(readFileSync(batchesPath, 'utf8'));
const written = [];
for (const batch of batches.batches) {
  const input = {
    projectRoot: 'D:/kodlar/arealai_case_study',
    batchFiles: batch.files,
    batchImportData: batch.batchImportData,
  };
  const out = `${tmpDir}/ua-file-analyzer-input-${batch.batchIndex}.json`;
  writeFileSync(out, JSON.stringify(input, null, 2));
  written.push(out);
}
console.log(written.join('\n'));
