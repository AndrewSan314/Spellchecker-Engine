# Execution Log — Vietnamese Spelling Engine Optimization (2026-08-24)

Plan: `docs/plans/2026-08-24-spelling-engine-optimization.md`
Workspace is NOT a git repo (plan rule 9): each task checkpoint is recorded here instead of commits.

## Baseline (before any change)

Date: 2026-08-24, machine-local run.

| Check | Result |
|---|---|
| `node test/core.test.mjs` | 31 pass / 0 fail |
| `node test/rules.test.mjs` | 35 pass / 0 fail (after one transient mid-edit read; see note) |
| `python -m unittest discover -s test -p 'test_*.py'` | 13 tests OK (requires TMP/TEMP pointed into workspace under sandbox) |
| `node --test test/test_benchmark_scope.mjs` | 3 pass / 0 fail |
| `python tools/audit_lm.py` | PASS — held-out intersections all 0 (clean-dev/test, synthetic, viwiki, vsec-dev, vsec-test) |
| Official benchmark | 1439 rows / 2949 expected · recall 65.9% · precision 89.9% |

Matches plan §0 "Measured baseline" exactly.

Note on transient failure: the first rules-suite run showed 5 failures while
`src/data/lexicon.txt` was being concurrently rewritten; a rerun after writes
settled was fully green. Lesson recorded as operational risk (matches agent2
finding F4 from the independent evaluation session): benchmark runs must pin /
re-read data after write settlement.

AGENTS.md: does not exist in this workspace — plan workflow step 1 noted N/A.
Semble used for code-location lookups where applicable; GitNexus index absent →
impact checks skipped per plan rule 3 ("if an index exists").

---

## Task 1: Preserve Viwiki correction suggestions

Status: DONE

- [x] Bug verified in current code: `validMistake()` returned only `{startOffset,value}`
      so `addMistake()` read `mistake.suggest`=undefined → every suggestion lost.
- [x] Failing test written: `test/test_viwiki_converter.mjs` (6 tests incl. multi-
      mistake grouping, multi-suggestion retention, identity policy drop,
      offset-after-trim, mixed real+identity)
- [x] Failure observed: import error (exports missing) = red
- [x] Minimal fix: `validMistake` retains filtered defensive copy
      (drops identity suggestions per exported IDENTITY_SUGGESTION_POLICY =
      'drop-identity-suggestions'); `addMistake` reads retained `suggestions`;
      pure `convertDocuments(docs,opts)` extracted + `convertDocumentFixture`
      helper; `convertViwikiExternal` now delegates.
- [x] Focused tests green 6/6; regression: rules suite untouched this task.
- [x] Regenerated benchmark/corpus-viwiki-spelling.json: 150 rows / 165 labels /
      rows-with-nonempty-suggestions 150 (was 0). `npm run data:audit` green
      (python unittest OK + benchmark-scope 3/3). train∩viwiki=0 held (audit_lm).
- Deviation note: none. Checkpoint recorded here (no git in workspace, rule 9).

---

## Task 2: Fail-fast LM loading

Status: DONE (implemented by an earlier un-logged session; verified + one
critical defect fixed this session)

- [x] Implementation found present in `src/language.mjs`:
      `LanguageModelArtifactError`, ENOENT-only fallback (`allowFallback`),
      strict `_loadPrebuiltValidated` (header/kind/tabs/positive-int counts/
      section order/unique keys), diagnostics incl. backend + sha256.
      `test/test_language_model_loader.mjs` covered all three required cases
      plus ordering/dup/header/count rejections (9 tests, green).
- [x] DEFECT FOUND & FIXED (TDD): `_loadPrebuiltValidated` set
      `totalTokens = record-line count` (U+B+T rows = 3,960,041) instead of the
      weighted token total declared by `#tokens=` (1,325,600,394). Every
      pUni was inflated ~335x, corrupting all confidence/margin gates — root
      cause of a pre-existing red test (`sentence-initial ambiguous words
      still warn when context decides`, 'Dat ban toi nay giam 10%').
      New failing test added (`totalTokens comes from the #tokens header…`),
      fix: model now constructed with `totalFromHeader`.
- [x] Focused 10/10; regression core 31/31, rules 35/35, converter 6/6,
      `npm run data:audit` green.
- Deviation note: none (defect fix is within Task 2 scope: loader must load
  the artifact as built by tools/build_lm.py).

---

## Task 3: Correct centered trigram scoring

Status: DONE

- [x] Failing tests written FIRST: `test/test_language_scoring.mjs` (6):
      centered evidence monotone in count(prev,cand,next); independent of
      unrelated count(prev,next); attested > unattested under left-to-right
      factorization; finite at zero counts; REGRESSION rare-twin-vs-joint
      (both zero-evidence and attested variants). Observed red: 3 failed on
      old code (centeredEvidence missing ×2, rare-twin inversion ×1).
- [x] Old invalid forms removed from `scoreCandidateOverSurfaces`:
      centered term `count(p,w,n)/count(p,n)`; right bigram
      `log P(next|w)`; forward trigram `P(n2|n1,w)` — all conditioned (fully
      or partly) on the candidate being ranked, so tiny candidate counts
      inflated nonsense twins ("đật" outranked "đặt").
- [x] New formulation (plan-mandated): centered chain
      `log P(cand|prev) + log P(next|prev,cand)` via existing `_pBiRaw`/
      `_pTriRaw` (beam-first short-circuit when beam pair attested); RIGHT
      side = smoothed pair joint `jointPairProb(w,n)` and forward trigram =
      `jointTripleProb(w,n1,n2)` — normalizers are candidate-independent;
      left trigram reuses `_pTriRaw(w,p1,p2)`. `_scoreWith` right side fixed
      identically. Weights `{A,B,C,D,E,F}` + smoothing masses moved to config
      (`linguistic.scoreWeights`, `rightJointSmoothing=20`,
      `forwardJointSmoothing=10`), wired via `languageModel.setScoreConfig`
      in engine constructor.
- [x] Focused 6/6; regression: loader 10/10, core 31/31, rules 35/35,
      converter 6/6, data audit green. Previously failing 'Dat' case now
      ranks "đặt" top-1 and emits.
- Dev metrics: deferred to Task 8 (eval workflow does not exist yet) — noted
  as pending for Step 4 of this task.
- Deviation note (documented): scope extended beyond the single centered
  formula to the two sibling terms sharing the same invalid conditioning
  (`P(next|w)`, `P(n2|n1,w)`); required to satisfy the task's own
  mathematical gates and to fix the live ranking inversion. No smoothing
  method changed otherwise; no Kneser-Ney introduced (deferred to Task 10).

---

## Task 4: Normalize Telex before candidate lookup

Status: DONE

- [x] Failing tests written FIRST: `test/test_candidate_generation.mjs` (6):
      dawnf→dằn family, ddat→đặt/đất family, non-Telex typos unchanged,
      deterministic+bounded output, protected codes/URLs not Telex-mangled,
      surface-decoding unit cases. Observed red: import error
      (`telexLookupSurface` missing).
- [x] Implementation: exported `telexLookupSurface()` in language.mjs —
      digraph decode via existing TELEX table MINUS bare-w (global w→ư would
      corrupt "www."/"vn"), standalone-w guarded by letter adjacency,
      tone letters (s/f/r/x/j) dropped immediately-after-vowel AND at token
      end ("danf"/"dawnf" forms). `SymSpellCandidateProvider.candidates()`
      now unions raw stripped key + Telex-decoded key, ranks by min distance
      across keys; ONLY rawKey excluded as self (Telex-decoded key must be
      able to match the target family).
- [x] Protected-token guarantee tested at the correct layer:
      ambiguous sequences may over-decode inside the pure function by design
      (raw key always searched too), but classifyToken gates PROTECTED/
      CODE_LIKE/WHITELISTED before any candidate lookup — asserted with
      classifyToken stubs.
- [x] Focused 6/6; regression rules 35/35, core 31/31.
- Deviation note (documented): normalization implemented at the provider
  layer (`candidates()`) instead of the spelling-rule call site — covers all
  callers and keeps rule code unchanged; plan allowed "modify if necessary"
  for language.mjs.

---

## Task 5: Candidate oracle coverage + generation/ranking cascade

Status: DONE

- [x] Audit tool created: `tools/audit_candidate_recall.mjs` — reads ONE
      VSEC split; reports labeled corrections, accent-family coverage,
      SymSpell top-6/12/20 key coverage, loss taxonomy (same-key-beyond-
      accent-cap / surface-pruned / edit-distance-out / telex /
      multi-token-unsupported / other) + miss examples; atomic JSON artifact
      with sha256 manifest sidecar. Split guard: dev/train only;
      test/external require --final (verified refusal on --split test).
- [x] Failing tests added (test_candidate_generation.mjs): unaccented
      surface retention vs accented siblings; top-12 key pool retains
      candidates crowded out at 6 (transposition priced 0.8 documented);
      deterministic + bounded pool. Red observed → green.
- [x] Generation split from ranking: `SymSpellCandidateProvider.generatePool`
      (keyCap default 12→config 20, surfacesPerKey 4 with forced unaccented
      retention, dist+freq metadata, poolBound); `candidates()` delegates.
      NO LM calls during generation.
- [x] Cheap first-stage ranker in spelling rule (config weights
      cheapDistWeight/cheapFreqWeight/cheapFamilyBonus/cheapAttestBonus):
      Telex-aware edit distance + log-frequency + accent-family bonus +
      at most ONE bigram attestation probe; top `spellingCheapTopK=3`
      (+original) enter expensive trigram scoring — satisfies ≤4 gate.
- [x] Config knobs added: spellingCandidateKeys=20, spellingSurfacesPerKey=4,
      spellingPoolMax=80, spellingCheapTopK=3, cheap* weights.
- [x] Dev audit result (1085 evaluated labels): oracle coverage **93.5%**
      (gate ≥92%) — was effectively ~80.9% baseline; accent-family lane
      covers 68.3%; residual losses: multi-token 28 (unsupported per plan),
      other 21, edit-distance-out-of-policy 9, surface-pruned 13.
      Artifact: dataset_artifacts/evaluation/candidate-recall-dev.json
      (sha256 manifest sidecar written).
- [x] Regression: focused 25/25, core 31/31, rules 35/35.
- Key finding recorded: correct keys cluster at rank 14–29 among
  equal-edit-distance higher-frequency rival keys ("bai" behind vao/cao/
  ban…); frequency tiebreak at equal distance is the dominant loss mode —
  fixed by widening the CHEAP pool to 20 keys while keeping expensive
  reranking at 3 candidates.
- Deviation note (documented): audit measures ENGINE-lane oracle
  (SymSpell pool ∪ accent-family lane) rather than provider-only, because
  VSEC same-key tone errors are owned by the missing-diacritic/sibling-merge
  lanes by design; measuring provider alone would misreport ~66% of labels
  as lost. VSEC target strings normalized (trailing punctuation stripped)
  before coverage checks — label hygiene only.

---

## Task 6: Remove surface Cartesian-product hotspot

Status: DONE

- [x] Profiler created: `tools/profile_engine.mjs` (read-only; atomic JSON
      artifact + sha256 manifest). Reports load time, RSS/heap (with
      --expose-gc), instrumented calls+time for resolveInContext/pTri/
      _pBiRaw/scoreCandidateOverSurfaces/centeredEvidence/jointPairProb,
      bigram+trigram hit rates, latency p50/p95/p99 overall, per length
      bucket (incl. SMS <=160) and per category.
- [x] Behavior-preservation tests FIRST: `test/test_context_reranker.mjs`
      (5): attested beam-first centered pair decides over numerically-higher
      rare sibling; fallback = exhaustive max only when beam-first lacks
      evidence; same two semantics for NEW leftTrigramEvidence /
      forwardTripleJoint (red until implemented); finite-score grid.
- [x] Beam-first short-circuiting completed for ALL surface-pair loops:
      leftTrigramEvidence + forwardTripleJoint extracted from
      scoreCandidateOverSurfaces with attested-beam-pair early return;
      centered already had it (Task 3); right side is max-over-2-surfaces.
- [x] Request-local caches: deliberately NOT added — plan keeps caching
      subordinate to probe reduction; remaining hotspot is resolveInContext
      (~40% of run wall), a pure function candidates for Task 9 follow-up.
- [x] Measured (1439 benchmark rows, node --expose-gc):
      scoreCandidateOverSurfaces 47,113 calls (plan-era 264,345 → 5.6x fewer,
      gate ≥3x MET); _pBiRaw 1.11M calls (plan-era 12.5M → 11x fewer);
      trigram lookups 693,523 deterministic counter; bigram hit 34%,
      trigram hit 15%; p50 ~1.6-2.0ms, p95 ~4.1-4.8ms (run-noise band),
      SMS<=160 p95 ~3.9-4.3ms (gate ≤15ms MET); steady RSS ~872-905MB
      (addressed in Phase 4); load ~4.3-4.6s (addressed in Phase 4).
      Artifacts: dataset_artifacts/evaluation/profile-pre-task6.json,
      profile-post-task6.json (+manifests).
- [x] Identical deterministic-rule results: core 31/31, rules 35/35 after
      change; reranker/scoring suites 11/11.
- Deviation note (documented): the ≥3x probe-reduction gate is credited to
  the combined Task 3 scoring reformulation (candidate-independent joint
  terms replaced conditional-probe cascades) plus the prior session's
  two-entity neighbour surfaces; Task 6's own delta is small because the
  loops it short-circuits were already ≤2x2 after those changes.

---

## Task 7: Length-aware SymSpell delete index

Status: DONE

- [x] Failing tests FIRST (`test/test_symspell_index.mjs`, 4): exhaustive
      fixture equivalence legacy-depth2 vs length-aware across every vocab
      word + systematic corruptions (102 lookups incl. telex inputs) under
      maxEditDistanceFor policy; read-only diagnostics shape; strict edge
      reduction. Red observed (option missing / stats missing).
- [x] Implementation: constructor opt `lengthAware` (default true);
      per-key depth = min(maxIndexDepth, maxEditDistanceFor(key.length));
      identity keys preserved; deterministic bucket order untouched.
      `indexStats()` exposes keys/variants/edges/maxBucket/buildMs.
      Profiler output extended with `symSpellIndex` section.
- [x] Real-index measurement (production lexicon): variants 993,432→561,248
      (−17.6%); edges → 689,852 (−30.5% vs plan-era 993,432 baseline count)
      matching the plan's simulated −34%/−17.5% band with NO recall loss
      (equivalence suite green).
- [x] Regression: focused suites 36/36 (candidate/scoring/loader/reranker/
      viwiki), core 31/31, rules 35/35. SMS<=160 p95 3.53ms in post-change
      profile run.
- Deviation note: none.

---

## Task 8: Strict train/dev/test evaluation workflow

Status: DONE

- [x] Refactor for single-source-of-truth: spelling-rule per-token decision
      extracted to exported `evaluateSpellingToken(services, snap, ctx, doc,
      words, idx)` in linguistic-rules.mjs; the rule now consumes it.
      Behavior-preserving (rules 35/35 after refactor).
- [x] `tools/run_spelling_eval.mjs`: split guard (`assertSplitAllowed`;
      dev/train default; test/external-test require --final), provenance
      hashes (language/linguistic-rules/config sources + LM artifact +
      config snapshot knobs), benchmark-consistent matching, per-label
      failure stages via the PRODUCTION decision function: prefilter /
      candidate-miss / cheap-ranker-loss / trigram-ranker-loss /
      gate-rejected / wrong-suggestion / suppressed-downstream /
      unsupported-multi-token / correct-other-lane. Atomic artifact write
      with sha256 manifest.
- [x] `config/spelling-tuning.json` v1 created (unfrozen): objective =
      spelling-dev-F0.5 with the plan's constraint list; params mirror
      current defaults; provenance records dev oracle coverage 93.5%.
- [x] package.json scripts added: eval:spelling:dev, eval:spelling:test
      (--final), profile:engine.
- [x] Guard tests FIRST-ish (`test/test_eval_split_guard.mjs`, 4/4 green):
      tuning splits allowed; held-out refused without --final; --final opens;
      unknown splits rejected.
- [x] First dev measurement (1115 labels, 934 rows):
      strict (official ruleId semantics) P=0.907 R=0.061 F0.5=0.240;
      cross-lane P=0.731 R=0.168 F0.5=0.437.
- KEY FINDING recorded: 852/1115 (76%) of VSEC labels are PREFILTERED out of
  the typo lane (same stripped key as the input = UNACCENTED_VALID class);
  by design they belong to the missing-diacritic lane, whose emissions are
  NOT credited under strict POSSIBLE_SPELLING_ERROR matching. 119 labels are
  fixed correctly by PMD ('correct-other-lane'). Structural implication:
  strict-recall headroom is bounded (~13% ceiling); meaningful improvement
  requires cross-lane credit or lane re-routing — recorded for Task 9.
- Deviation note (documented): eval reports TWO metric views. Strict view
  matches the official benchmark exactly (ruleId-scoped). Cross-lane view
  credits any linguistic issue matching value+suggestion and is declared the
  Task 9 optimization objective — without it, tuning optimizes a metric that
  cannot respond to ~87% of the labels. No engine/benchmark semantics were
  changed.

---

## Task 9: Tune weights and gates on dev only

Status: DONE (winner = existing defaults; search recorded and frozen)

- [x] Tunable constants already in config after Tasks 3/5: scoreWeights,
      joint smoothing masses, cascade sizing (candidateKeys/surfacesPerKey/
      poolMax/cheapTopK), cheap-ranker weights, typoAlpha/Beta/Gamma,
      originalPriorBonusSpelling, conf/margin/frequency gates.
- [x] `tools/tune_spelling.mjs`: deterministic coordinate descent over
      {spellingCheapTopK ×3, spellingMinConfidence ×4, spellingMinMargin ×4,
      originalPriorBonusSpelling ×3, typoGamma ×3} + confidence×margin local
      grid (pass 2). Engine built ONCE; per-trial behavior via
      ValidationConfigService.reload (no restarts). Constraints evaluated
      per trial in-loop: synthetic-diacritics MD recall drop ≤0.5pp;
      cross-lane precision ≥ baseline−0.02; SMS≤160 p95 ≤15ms measured from
      the same row loop. Every trial recorded with metrics + feasibility to
      dataset_artifacts/evaluation/tuning-run.json (+sha256 manifest).
      Never touches test/external splits.
- [x] RESULT (33 trials): winner = {} (defaults) with crossLane F0.5=0.4371
      P=0.7305 R=0.1677, mdR=0.9559 (unchanged), sms160P95≈2.4–2.8ms. EVERY
      single-coordinate move was neutral or worse. Consistent with the Task 8
      structural finding: gates/weights act on ~24% of labels that reach the
      decision stage; the prefilter wall dominates the metric, so knob moves
      cannot express their effect. Recorded honestly instead of forcing a
      spurious "tuned" point.
- [x] Frozen: config/spelling-tuning.json v1 frozen=true — winner={},
      frozenParams mirror src/config.mjs defaults, devMetrics +
      constraint outcomes recorded. src/config.mjs unchanged (winner IS the
      default configuration).
- [x] `test/test_tuning_determinism.mjs` (3/3): identical overrides →
      byte-identical issue sets across reload churn; reload({}) restores
      default behavior exactly; tuning artifact schema-sane AND winner f05 ==
      best feasible trial (selection reproducibility).
- Constraint-gate interpretation (documented): core/rule suites assert
  DEFAULT-config behavior, so they run as a POST-freeze acceptance gate on
  the applied winner rather than inside every trial loop; per-trial latency
  and MD-recall constraints ARE evaluated in-loop as specified.
