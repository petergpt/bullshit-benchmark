/* Exercise the report's real scoring and rendering code without service calls. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const template = fs.readFileSync(path.join(__dirname, '../scripts/report_template.html'), 'utf8');
const script = template.match(/<script>([\s\S]*?)<\/script>/)[1];

function report() {
  const nodes = new Map();
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, { value: '', innerHTML: '', textContent: '', querySelectorAll: () => [] });
      return nodes.get(id);
    },
  };
  const context = vm.createContext({ document });
  vm.runInContext(script.replace('__REPORT_PAYLOAD__', JSON.stringify({ judge_runs: [{}, {}, {}] }))
    .replace(/\bboot\(\);\s*$/, ''), context);
  return { context, nodes };
}

function scored(score = 2, changes = {}) {
  return {
    model: 'example/model', sample_id: 'sample', question_id: 'q', run_index: 1,
    consensus_score: score, status: 'ok', response_text: 'An answer.',
    judges: ['a', 'b', 'c'].map(model => ({ model, score, error: '', status: 'ok' })),
    ...changes,
  };
}

function missing(changes = {}) {
  const row = scored(2);
  row.judges[2] = { model: 'c', score: null, error: 'content_filter', status: 'error' };
  return { ...row, consensus_score: null, ...changes };
}

function approvedPartial(changes = {}) {
  const row = missing({
    consensus_score: 1, consensus_method: 'mean',
    judge_valid_count: 2, judge_expected_count: 3,
    judge_coverage: '2/3 judges', judge_failure_policy: 'retry_then_two_valid',
  });
  row.judges[0].score = 0;
  row.judges[2].attempt_count = 3;
  return { ...row, ...changes };
}

test('null and blank values remain unscored, while real zero is preserved', () => {
  const { context: c } = report();
  for (const value of [null, undefined, '', ' ', false]) {
    assert.equal(c.toNumberOrNull(value), null);
    assert.equal(c.fmtPct(value), 'n/a');
  }
  assert.equal(c.consensusBucket(null), null);
  assert.equal(c.fmtPct(0), '0%');
  assert.equal(c.rowConsensusScore(scored(0)), 0);
  assert.equal(c.getRowState(scored(0)).outcome.key, 'bs_missed');
});

test('failed or absent judges cannot be replaced by stale consensus or remaining votes', () => {
  const { context: c } = report();
  const shortPanel = scored(2, { judges: scored(2).judges.slice(0, 2) });
  const noAggregate = missing();
  delete noAggregate.consensus_score;
  const historical = scored(0);
  historical.judges[2].justification = 'Fallback score: judge returned empty output after retries.';
  for (const row of [
    missing(), missing({ consensus_score: 2 }), noAggregate, shortPanel, historical,
    scored(2, { status: 'error' }), scored(2, { error: 'collection failed' }),
    scored(2, { row_errors: ['missing evaluation'] }),
    scored(2, { consensus_error: 'incomplete_judge_panel' }),
    scored(2, { consensus_score: null }),
  ]) {
    assert.equal(c.rowConsensusScore(row), null);
    assert.equal(c.getRowState(row).pending, true);
    assert.equal(c.getRowState(row).consensus, null);
  }
});

test('valid completed scores, successful retries, and target refusals keep their meanings', () => {
  const { context: c } = report();
  const recovered = scored(0);
  recovered.judges[2].warnings = ['judge_raw_text_empty', 'judge_retry_on_empty=1'];
  recovered.judges[2].finish_reason = 'stop';
  assert.equal(c.rowConsensusScore(recovered), 0);
  assert.equal(c.getRowState(recovered).pending, false);
  const mixed = scored(1.6667, { judges: scored(2).judges });
  mixed.judges[0].score = 1;
  assert.equal(c.getRowState(mixed).consensus, 2);
  const refusal = scored(null, { response_refusal: true, judges: [] });
  assert.equal(c.getRowState(refusal).outcome.key, 'refusal');
  assert.equal(c.getRowState(refusal).pending, false);
  const board = c.computeBoard([scored(0), refusal])[0];
  assert.equal(board.refusalRate, 0.5);
  assert.equal(board.missedRate, 0.5);
  assert.equal(board.pending, 0);
});

test('incomplete models have coverage and no headline rates or extra zero scores', () => {
  const { context: c } = report();
  const board = c.computeBoard([scored(2), missing(), scored(0, { model: 'example/complete' })]);
  const pending = board.find(row => row.model === 'example/model');
  assert.equal(pending.pending, 1);
  assert.equal(pending.complete, 1);
  assert.equal(pending.correct, 1);
  assert.equal(pending.missed, 0);
  for (const key of ['correctRate', 'partialRate', 'missedRate', 'invalidRate', 'refusalRate', 'agreementRate', 'controlCorrectRate']) {
    assert.equal(pending[key], null, key);
  }
  const complete = board.find(row => row.model === 'example/complete');
  assert.equal(complete.correctRate, 0);
  assert.equal(complete.missedRate, 1);
});

test('overview visibly marks incomplete headline results and model coverage as pending', () => {
  const { context: c, nodes } = report();
  const rows = [scored(2), missing()];
  c.renderHeaderMeta(rows);
  c.renderOverview(rows);
  assert.match(nodes.get('headerMeta').textContent, /complete 1\/2 \| pending 1/);
  assert.match(nodes.get('kpiGrid').innerHTML, /BS Correct<\/div>\s*<div class="kpi-value">Pending<\/div>/);
  assert.match(nodes.get('overviewChart').innerHTML, /complete 1\/2/);
  assert.match(nodes.get('overviewChart').innerHTML, /pending 1 row/);
  assert.match(nodes.get('overviewChart').innerHTML, /class="ov-pct">Pending<\/span>/);
  assert.doesNotMatch(nodes.get('overviewChart').innerHTML, /correct 50%|correct 100%/);
});

test('response cards show the missing judge and pending evaluation rather than score zero', () => {
  const { context: c, nodes } = report();
  c.renderCompare([{ key: 'group', question_id: 'q', run_index: 1, rows: [missing()] }]);
  const html = nodes.get('compareGrid').innerHTML;
  assert.match(html, /Missing judge \| score pending/);
  assert.match(html, /c:missing err/);
  assert.match(html, /Pending evaluation \(2\/3 judges\)/);
  assert.doesNotMatch(html, /c:0|score 0/);
});

test('approved exhausted partials average two genuine votes and retain explicit coverage', () => {
  const { context: c, nodes } = report();
  const row = approvedPartial();
  assert.equal(c.rowConsensusScore(row), 1);
  assert.equal(c.getRowState(row).pending, false);
  assert.equal(c.getRowState(row).coverage.acceptedPartial, true);
  assert.equal(c.getRowState(row).allPresent, false);
  const board = c.computeBoard([scored(2), row])[0];
  assert.equal(board.pending, 0);
  assert.equal(board.correctRate, 0.5);
  assert.equal(board.partialRate, 0.5);
  c.renderCompare([{ key: 'group', question_id: 'q', run_index: 1, rows: [row] }]);
  const html = nodes.get('compareGrid').innerHTML;
  assert.match(html, /2\/3 judges · mean of valid grades/);
  assert.match(html, /a:0/);
  assert.match(html, /c:missing err/);
  assert.doesNotMatch(html, /Pending evaluation|c:0/);
});

test('partial labels do not authorize stale scores, absent votes, or unproven retries', () => {
  const { context: c } = report();
  const unattempted = approvedPartial();
  unattempted.judges[2].attempt_count = 0;
  const missingRow = approvedPartial();
  missingRow.judges[2].row_present = false;
  const badAttempts = approvedPartial();
  badAttempts.judges[2].attempts = [{}, {}, {}];
  const oneVote = approvedPartial();
  oneVote.judges[1].score = null;
  const absent = approvedPartial();
  absent.judges.pop();
  for (const row of [
    approvedPartial({ consensus_score: 0.6667 }),
    approvedPartial({ consensus_score: null }),
    approvedPartial({ judge_failure_policy: undefined }),
    approvedPartial({ judge_expected_count: 2 }),
    approvedPartial({ row_errors: ['Identity mismatch'] }),
    approvedPartial({ error: 'Collection failure' }),
    unattempted, missingRow, badAttempts, oneVote, absent,
  ]) assert.equal(c.rowConsensusScore(row), null);
  assert.equal(c.rowConsensusScore(scored(1)), 1);
  assert.equal(c.rowConsensusScore(scored(0, { judges: scored(2).judges })), null);
});
