import { runEngineTests } from './engine.test.ts';

const { results, summary } = runEngineTests();

console.log('=============================================');
console.log(`TOTAL TESTS: ${summary.total}`);
console.log(`PASSED:      ${summary.passed}`);
console.log(`FAILED:      ${summary.failed}`);
console.log(`DURATION:    ${summary.totalDurationMs} ms`);
console.log('=============================================');

for (const r of results) {
  const status = r.passed ? '  PASS' : '  FAIL';
  console.log(`${status} [${r.category}] ${r.name} (${r.durationMs}ms)`);
  if (!r.passed) {
    console.error(`  ERROR: ${r.error}`);
  }
}

if (summary.failed > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
