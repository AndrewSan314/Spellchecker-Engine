// Task 2 helper — materialize golden `expected` blocks + JS-derived `units`
// into test/fixtures/attention-tokenizer-cases.json from the JavaScript
// reference implementation. Run after any INTENTIONAL tokenizer contract
// change, then review the diff carefully.
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ValidationContext } from '../src/engine.mjs';
import { buildValidationDocument } from '../src/document-builder.mjs';
import {
  unitsFromDocument, encodeContextUnits, encodeOptionSurface,
} from '../src/attention-tokenizer.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const FIXTURE_PATH = path.join(ROOT, 'test', 'fixtures', 'attention-tokenizer-cases.json');

const fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8'));
const VOCAB = new Map(Object.entries(fixture.vocabSample));

const docOf = (text) => buildValidationDocument(
  new ValidationContext(text, 'ACCENTED', 'TENDOO'));

for (const c of fixture.cases) {
  if (c.optionSurfaces) {
    c.expectedOptions = c.optionSurfaces.map((s) => encodeOptionSurface(s, VOCAB));
    continue;
  }
  const text = c.text ?? c.textNfc;
  const units = unitsFromDocument(docOf(text));
  c.units = units;
  const out = encodeContextUnits(units, c.targetWordIndex, VOCAB);
  // NFD/NFC pair: mirror the materialized vector onto the partner case,
  // proving byte-identical outputs across normalization forms
  if (c.pairCase) {
    const partner = fixture.cases.find((x) => x.name === c.pairCase);
    partner.text = text.normalize('NFD');
    const pUnits = unitsFromDocument(docOf(partner.text));
    partner.units = pUnits;
    partner.expected = encodeContextUnits(pUnits, partner.targetWordIndex, VOCAB);
    c.expected = out;
    continue;
  }
  c.expected = out;
}

writeFileSync(FIXTURE_PATH, `${JSON.stringify(fixture, null, 2)}\n`, 'utf8');
console.log(`materialized ${fixture.cases.length} cases -> ${path.relative(ROOT, FIXTURE_PATH)}`);
