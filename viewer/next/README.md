# BullshitBench dashboard

Entry point: `viewer/index.next.html`. The site root, `viewer/index.html` and
`viewer/index.v2.html` redirect here; `viewer/index.legacy.html` retains the legacy viewer. Both read the same published
datasets. No build step or package installation is needed.

Serve the repository root with `python3 -m http.server 8795 --bind 127.0.0.1`,
then open `http://127.0.0.1:8795/viewer/index.next.html`.

- **Dashboard:** one full-width outcome chart with thick bars, local company logos and a separate reasoning column. **Columns** adds accepted-premise rate, average grade (out of 2), total/reasoning/output tokens and cost; all are off by default. **Filters → Sort** includes Least accepted and Average grade, with the relevant column shown automatically. Rank numbers remain clear-pushback ranks. The original mascot and forest palette are retained, with a Futura Medium headline where the font is available and a sans-serif fallback. Domain remains a filter and model detail, with no dedicated chart tab.
- **Labs:** one compact company picker filters every view, with local logos beside every company and the selected name in the toolbar. The two-column menu supports arrows, Home/End, typing a company name, Enter, Escape and Tab. All labs restores the comparison; choices remain available when filtered. Reasoning and Best per model live in the shared Filters menu on desktop and mobile.
- **Timeline:** releases by date, with lab or detection colouring. Size uses the same lab filter and colour control; there are no duplicate lab strips or legends.
- **Lab trends:** OpenAI, Anthropic and Google release histories, using the legacy viewer's highest clear-pushback result per lab and exact release date. Lines join those releases (not a cumulative maximum); logo summaries identify each lab's latest plotted model. By default, solid lines exclude refusals and faint dotted lines show the all-attempt rate of the same selected models. Only labs with differing rates get a dotted comparison; it adds no duplicate points or labels and can be hidden with the **All attempts** checkbox below the chart. Latest-model summaries and tooltips add the other rate only when different. Tooltips omit zero-refusal lines. Shared filters and V1/V2 switching recompute the winners, and `?view=explorer&chart=labs` opens the view directly.
- **Reasoning:** matched lowest/highest requested efforts in green Higher and red Lower panes, with company logos. Equal results are neutral and available in the Unchanged foldout. Tooltips show provider-reported reasoning and output tokens, with coverage when incomplete. Missing telemetry is unknown, not zero; similar means alone do not establish that a provider ignored the requested effort.
- **Size:** company or detection colouring.
- **Responses:** outcome/domain drilldowns, question search and independent model selectors; A/B tabs on phones. Selected answers and questions stay within the page and are not added to the URL.

UI notes stay brief: omit repeated rates and zero counts, show judge coverage as a count, and keep the method in **Scoring**.

Charts use measured labels that avoid overlaps, prioritizing highlighted models. Lab trends also reserves label slots for each lab's latest plotted release and searches farther for a clear placement. Latest model names wrap in the summary so they remain readable even when the plot is too small for every label.
Table values use a shared regular type scale, with medium weight for highlighted rows. Filter and column menus stay within the viewport and scroll on short screens; phone controls use two compact rows.
An inline **New** label marks model variants for exactly seven days from their first recorded benchmark test. It stays on the model-name line in the Dashboard and PNG exports, and expires automatically in an open viewer. Recency uses each suite's first-test metadata, not model release dates or the latest dataset refresh.

The dashboard's **Exclude refusals** checkbox starts unticked and controls bar composition only. Clear scores, rankings, domain/question summaries, chart points and averages always exclude candidate refusals. Errors remain in rate denominators. Refusal rates and the optional all-attempt comparison use all attempts. Tooltips and model details show both clear rates when refusals change the result. The legacy viewer retains its separate denominator control. Canonical aggregates and published leaderboard CSVs retain their all-attempt rates.

CSV exports keep `rate_denominator`, `clear_pushback_pct` and `exclude_refusals` fixed to non-refusal scoring, alongside explicit all-attempt and non-refusal rates. `bars_exclude_refusals` and `bar_denominator` record the checkbox separately. Reasoning/output token averages and measurement counts are included. An all-refusal group displays an unavailable primary rate and is omitted from plots.

Click model names or their checkboxes to toggle multiple highlights. **Highlighted
only** preserves ranks within the other active filters. Highlighting and filters
are retained in the URL. The pin button hides navigation/filter chrome and keeps
a clean benchmark header above the scrolling chart; Escape exits this mode.
**PNG** exports the fully visible bar rows at 2× resolution, with logos, highlights,
filter labels and rank/excerpt context. Clear percentages always exclude refusals; bars follow the checkbox. One footer line records both denominators and any reduced judge coverage. PNG export is available on the
Dashboard; other chart views retain normal browser screenshots. Optional table
columns do not change the outcome chart exported by PNG or the full CSV fields.

Company assets are stored locally; see `assets/brands/SOURCES.md` for the pinned
source revisions, exact URLs, licenses and hashes. Anonymous/unknown providers use
an initial rather than an invented logo.

Both V1 and V2 use the published manifests and immutable assets. `data.mjs`
checks asset sizes, SHA-256 hashes, row counts and sample pairing before use.
Response text loads per question, with four concurrent fetches and a twelve-question
cache. Model output is escaped before a small Markdown subset is rendered.
The URL retains the suite, view, filters, highlighted models, pin state, chart, sorting and optional columns. The older `?version=v1` suite alias remains supported alongside `?benchmark=v1`. Model focus and answer selections are not serialized.
`columns` accepts `accepted`, `score`, `tokens`, `reasoning`, `output` and `cost`. Older
`?view=rankings` links open Dashboard with the original grade, tokens and cost columns.
Column changes preserve chart scroll; hiding the current sort column restores
Clear descending. Narrow screens scroll additional columns horizontally.
Generic response navigation and suite changes clear prior outcome drilldowns.

Files: `app.mjs` handles interaction; `app.css` handles layout; `charts.mjs`
renders native SVG and HTML; `labels.mjs` places chart labels without collisions;
`data.mjs` reads data without mutating it.
`brands.mjs` resolves local logos and original provider colours; `capture.mjs`
produces PNGs from the supplied rows without changing application state.

## Verification

Run `node --test tests/test_next_viewer_data.mjs tests/test_next_viewer_capture.mjs tests/test_published_viewer_storage.js tests/test_viewer_refusal_denominators.js`
from the repository root. The tests check all 32,070 published rows against both
viewers and their leaderboard CSVs: V1 has 10,670 answers / 194 variants; V2 has
21,400 answers / 214 variants. They also check response text, lazy loading,
concurrency, missing-judge selection and rejection of invalid assets. Refusal-denominator checks cover both rates, explicit inclusion, reset, retained error rows and unavailable rates for all-refusal groups.

The adapter follows immutable viewer rows. Each failed judge has three output attempts;
after exhaustion, the two valid votes are averaged. Missing votes remain
unscored, and fewer than two valid votes leave the answer unavailable. Affected
answers carry “2/3 judges”. Coverage remains in answer details, chart tooltips,
CSV exports and the PNG footer, without badges in model labels or Lab trends summaries.
Selecting an unavailable individual judge never substitutes
zero or the consensus result.

The V2 dataset contains 77 answers with two valid judges: 73 historical answers,
two GPT-6 Astra answers and two DeepSeek V4.1 Flash answers. Two candidate refusals
reviewed alongside the historical repair remain unscored. Real judge votes and model
responses are preserved. Both suites' leaderboard CSVs match viewer outcomes in All attempts mode,
including the two previously discrepant Fable 5 rows; no CSV exceptions remain.
Both viewers use the manifest's frozen question snapshot when present, with a
legacy fallback for publications that predate question snapshots.

Browser verification covers desktop (1440 × 900), tablet (768 × 1024), phones
(320 × 568 and 390 × 844) and landscape (844 × 390): suite switching, shared lab
filters, reasoning and best-model filters, reset, optional grade sorting, chart
selection, question switching, A/B answer loading and viewport-bounded menus.
Phone bars reclaim the duplicate percentage column while retaining Clear sorting.
PNG downloads were verified for visible/scrolled row excerpts and highlighted-only
selections, including preserved ranks, refusal exclusions and local logo rendering.
These are local release checks. Confirm the deployed commit and populated pages after
GitHub Pages deployment; local validation does not establish the live site's state.
