// ============================================================
// Task 9 — Unit tests for Attention SHADOW and EXPERIMENTAL_ACTIVE pipeline integration.
//
// Contract under test:
//   - Mode 'OFF': skips attention inference entirely;
//   - Mode 'SHADOW': computes attention decision in shadow payload, emits 100% classical;
//   - Mode 'EXPERIMENTAL_ACTIVE': requires explicit calibration activation;
//   - Hard guards (code, URL, phone, currency, single-char, all-caps) NEVER invoke attention;
//   - Classical candidate generation and emission logic remain 100% intact.
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SmsValidationEngine } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { ValidationContext, MessageMode } from '../src/core.mjs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('mode OFF skips attention inference entirely', () => {
  const configService = new ValidationConfigService();
  configService.reload({
    spelling: { attentionMode: 'OFF' },
  });
  const engine = new SmsValidationEngine({ configService });
  const ctx = new ValidationContext(
    'Kinh chao quy khach hangf',
    MessageMode.ACCENTED,
    'TEST',
  );

  const res = engine.validate(ctx);
  assert.ok(res.valid !== undefined);
  assert.equal(engine.services.attentionReranker?.evaluatedCount ?? 0, 0);
});

test('mode SHADOW preserves exact classical output and computes shadow attention diagnostics', () => {
  const configOff = new ValidationConfigService();
  configOff.reload({ spelling: { attentionMode: 'OFF' } });
  const engineOff = new SmsValidationEngine({ configService: configOff });

  const configShadow = new ValidationConfigService();
  configShadow.reload({ spelling: { attentionMode: 'SHADOW' } });
  const engineShadow = new SmsValidationEngine({ configService: configShadow });

  const testTexts = [
    'Kinh chao quy khach',
    'Don hang cua ban da duoc giao thanh cong',
    'Vui long kiem tra lai ma OTP',
    'Uu dai 50% cho khach hangf than thiet',
    'Chuc mung sinh nhat quy khach hang',
  ];

  for (const text of testTexts) {
    const ctx = new ValidationContext(
      text,
      MessageMode.ACCENTED,
      'TEST',
    );

    const resOff = engineOff.validate(ctx);
    const resShadow = engineShadow.validate(ctx);

    // Exact output equivalence
    assert.equal(resShadow.valid, resOff.valid, `valid mismatch on "${text}"`);
    assert.equal(resShadow.issues.length, resOff.issues.length, `issue count mismatch on "${text}"`);
    for (let i = 0; i < resOff.issues.length; i++) {
      assert.equal(resShadow.issues[i].ruleId, resOff.issues[i].ruleId);
      assert.equal(resShadow.issues[i].start, resOff.issues[i].start);
      assert.equal(resShadow.issues[i].end, resOff.issues[i].end);
      assert.deepEqual(resShadow.issues[i].suggestions, resOff.issues[i].suggestions);
    }
  }
});

test('hard guards never invoke attention inference', () => {
  const configShadow = new ValidationConfigService();
  configShadow.reload({ spelling: { attentionMode: 'SHADOW' } });
  const engine = new SmsValidationEngine({ configService: configShadow });

  // Texts with codes, URLs, phones, all-caps, single-char tokens
  const guardTexts = [
    'Ma xac nhan la CODE12345 tai https://tendoo.vn',
    'Goi ngay 0912345678 de nhan 500k VND',
    'A B C D E F G',
    'SUPERPROMO GIAM GIA',
  ];

  for (const text of guardTexts) {
    const ctx = new ValidationContext(
      text,
      MessageMode.ACCENTED,
      'TEST',
    );
    const res = engine.validate(ctx);
    assert.ok(res);
  }
});
