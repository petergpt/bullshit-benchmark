import assert from 'node:assert/strict';
import test from 'node:test';
import { exportRankingsPng } from '../viewer/next/capture.mjs';

test('PNG Clear excludes refusals while bars follow the checkbox, including empty denominators', async () => {
  const originalDocument = globalThis.document;
  const drawn = [];
  const context = new Proxy({
    fillText(value) { drawn.push(String(value)); },
    measureText(value) { return { width: String(value).length * 6 }; },
  }, { get(target, key) { return key in target ? target[key] : () => {}; } });
  globalThis.document = {
    baseURI: 'file:///benchmark/viewer/index.next.html',
    createElement() { return { getContext: () => context, toBlob: done => done(new Blob(['png'])) }; },
  };
  try {
    // A model already summarized with refusal exclusion must still draw all-attempt bars.
    const model = { id: 'example', name: 'Example', org: 'example', reasoning: 'xhigh', rank: 1,
      total: 100, rateRows: 89, green: 77, amber: 10, red: 2, refusal: 11 };
    await exportRankingsPng({ models: [model], excludeRefusals: false });
    assert.ok(drawn.includes('86.5%'), 'headline uses 77/89');
    assert.ok(drawn.includes('77%'), 'green bar uses 77/100');
    assert.ok(drawn.includes('xHigh'));
    assert.ok(drawn.some(text => text.includes('Bars: all attempts · Clear: excl. refusals')));

    drawn.length = 0;
    await exportRankingsPng({ models: [model], excludeRefusals: true });
    assert.ok(drawn.includes('86.5%'), 'headline remains unchanged');
    assert.ok(drawn.includes('87%'), 'green bar uses 77/89 rounded');
    assert.ok(!drawn.includes('77%'));

    for (const excludeRefusals of [false, true]) {
      drawn.length = 0;
      await exportRankingsPng({ models: [{ ...model, total: 1, rateRows: 0,
        green: 0, amber: 0, red: 0, refusal: 1 }], excludeRefusals });
      assert.ok(drawn.includes('—'), 'all-refusal model has no Clear score');
      assert.equal(drawn.includes('100%'), !excludeRefusals, 'refusal bar follows checkbox');
    }
  } finally { globalThis.document = originalDocument; }
});
