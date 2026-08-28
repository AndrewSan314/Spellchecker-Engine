// ============================================================
// Task 8 (tiny-attention-spelling-reranker-FIXED plan) — Pure JavaScript
// Typed-Array Inference Runtime for Tiny Attention Spelling Reranker.
//
// Constraints honored:
//   - Dependency-free ESM using Node.js built-ins (TypedArrays, DataView, fs);
//   - No PyTorch, no ONNX, no TensorFlow, no native addons, no new npm packages;
//   - Validates binary magic (TDRANK01), version 1, and SHA-256;
//   - Pre-LN Transformer single context pass (<=32 tokens) per target token;
//   - Shared option scoring head for [KEEP_ORIGINAL, cand_1, ..., cand_K];
//   - Microsecond-level batch-1 latency, deterministic numerical output.
// ============================================================
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';

const MAGIC_STR = 'TDRANK01';
const FORMAT_VERSION = 1;
const SQRT_2_OVER_PI = 0.7978845608028654;
const GELU_COEFF = 0.044715;

function gelu(x) {
  return 0.5 * x * (1.0 + Math.tanh(SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)));
}

function layerNorm(input, weight, bias, out, eps = 1e-5) {
  const len = input.length;
  let sum = 0;
  for (let i = 0; i < len; i++) sum += input[i];
  const mean = sum / len;

  let varSum = 0;
  for (let i = 0; i < len; i++) {
    const diff = input[i] - mean;
    varSum += diff * diff;
  }
  const invStd = 1.0 / Math.sqrt(varSum / len + eps);

  for (let i = 0; i < len; i++) {
    out[i] = (input[i] - mean) * invStd * weight[i] + bias[i];
  }
}

function matvecLinear(vec, weight, scales, bias, out) {
  // vec: Float32Array(in_dim)
  // weight: Int8Array(out_dim * in_dim)
  // scales: Float32Array(out_dim)
  // bias: Float32Array(out_dim)
  // out: Float32Array(out_dim)
  const inDim = vec.length;
  const outDim = out.length;

  for (let o = 0; o < outDim; o++) {
    const rowOffset = o * inDim;
    let dot = 0;
    for (let i = 0; i < inDim; i++) {
      dot += vec[i] * weight[rowOffset + i];
    }
    out[o] = dot * scales[o] + (bias ? bias[o] : 0);
  }
}

function addCharEmbedding(dest, hashes, charWeight, charScales, hiddenDim) {
  if (!hashes || !charWeight || !charScales || hashes.length === 0) return;
  const count = Math.min(hashes.length, charWeight.length / hiddenDim);
  for (let i = 0; i < count; i++) {
    const hash = hashes[i];
    if (!Number.isInteger(hash) || hash < 0 || hash >= charScales.length) continue;
    const off = hash * hiddenDim;
    const scale = charScales[hash];
    for (let d = 0; d < hiddenDim; d++) dest[d] += charWeight[off + d] * scale / count;
  }
}
function matvecFloat(vec, weight, bias, out) {
  // Float32 linear
  const inDim = vec.length;
  const outDim = out.length;

  for (let o = 0; o < outDim; o++) {
    const rowOffset = o * inDim;
    let dot = 0;
    for (let i = 0; i < inDim; i++) {
      dot += vec[i] * weight[rowOffset + i];
    }
    out[o] = dot + (bias ? bias[o] : 0);
  }
}

/**
 * Number of encoder blocks present in a checkpoint, counted by walking
 * `encoder.blocks.N.*` upward from 0 so a gap can never be read as a larger
 * model than actually shipped.
 */
function countEncoderBlocks(tensors) {
  let n = 0;
  while (tensors[`encoder.blocks.${n}.ln1.weight`]) n++;
  return n;
}

class AttentionRerankerInstance {
  constructor(metadata, tensors, buffer) {
    this.meta = metadata;
    this.arch = metadata.arch || 'A';
    this.k = metadata.k || 8;
    this.config = metadata.config || {};
    this.hiddenDim = this.config.hidden_dim || 48;
    this.numHeads = this.config.num_heads || 2;
    this.headDim = this.config.head_dim || 24;
    this.ffnDim = this.config.ffn_dim || 96;
    this.maxTokens = 32;
    // Phase 3: the forward pass used to hard-code `encoder.blocks.0.*`, so
    // any checkpoint with more than one block silently ran only its first
    // layer. That is why arch A shipped -- selection recorded it as
    // "runtime-compatible", not better (calibration F0.5: C 0.9535 vs A
    // 0.9370). Block count is read from the checkpoint itself so the tensors
    // stay the single source of truth; num_layers in config is a cross-check.
    this.numBlocks = countEncoderBlocks(tensors);
    const declaredBlocks = this.config.blocks
      ?? this.config.num_layers ?? this.config.num_blocks;
    if (declaredBlocks != null && declaredBlocks !== this.numBlocks) {
      throw new Error(`attention-reranker: checkpoint declares ${declaredBlocks}`
        + ` block(s) but carries tensors for ${this.numBlocks}`);
    }
    if (this.numBlocks < 1) {
      throw new Error('attention-reranker: no encoder.blocks.N.* tensors found');
    }

    this.tensors = tensors;
    this.buffer = buffer;

    // Pre-allocated scratch buffers for memory efficiency and speed
    this.scratch = {
      x: new Float32Array(this.maxTokens * this.hiddenDim),
      ln1_x: new Float32Array(this.maxTokens * this.hiddenDim),
      q: new Float32Array(this.maxTokens * this.hiddenDim),
      k: new Float32Array(this.maxTokens * this.hiddenDim),
      v: new Float32Array(this.maxTokens * this.hiddenDim),
      attnScores: new Float32Array(this.numHeads * this.maxTokens * this.maxTokens),
      attnOut: new Float32Array(this.maxTokens * this.hiddenDim),
      projOut: new Float32Array(this.maxTokens * this.hiddenDim),
      ln2_a: new Float32Array(this.maxTokens * this.hiddenDim),
      ffn1: new Float32Array(this.ffnDim),
      ffn1Gelu: new Float32Array(this.ffnDim),
      ffn2: new Float32Array(this.hiddenDim),
      finalH: new Float32Array(this.maxTokens * this.hiddenDim),
      tokenBuf: new Float32Array(this.hiddenDim),
      tokenOut: new Float32Array(this.hiddenDim),

      // Option scoring scratch
      classProj: new Float32Array(16),
      z: new Float32Array(this.hiddenDim * 3 + 16 + 4),
      sc1: new Float32Array(this.hiddenDim),
      sc1Gelu: new Float32Array(this.hiddenDim),
      sc2: new Float32Array(1),
    };
  }

  scoreOptions({
    wordIds,
    mask,
    targetPosition,
    optionWordIds,
    optionMask,
    classicalFeatures,
    markers = null,
    charHashes = null,
    optionCharHashes = null,
  }) {
    const H = this.hiddenDim;
    const s = this.scratch;
    const t = this.tensors;

    // 1. Embed tokens into scratch.x (32, 48)
    s.x.fill(0);
    const wordWeight = t['encoder.word_embedding.weight'].data;
    const wordScales = t['encoder.word_embedding.weight'].scales;
    const posWeight = t['encoder.position_embedding.weight']?.data;
    const posScales = t['encoder.position_embedding.weight']?.scales;
    const markerWeight = t['encoder.marker_embedding.weight']?.data;
    const markerScales = t['encoder.marker_embedding.weight']?.scales;
    const charWeight = t['encoder.char_embedding.weight']?.data;
    const charScales = t['encoder.char_embedding.weight']?.scales;

    for (let pos = 0; pos < this.maxTokens; pos++) {
      if (pos >= wordIds.length || mask[pos] === 0) continue;
      const wid = wordIds[pos];
      const rowOff = wid * H;
      const scale = wordScales[wid];

      const pOff = pos * H;
      const pScale = posScales ? posScales[pos] : 1.0;

      const destOff = pos * H;
      for (let d = 0; d < H; d++) {
        let val = wordWeight[rowOff + d] * scale;
        if (posWeight) val += posWeight[pOff + d] * pScale;
        if (markerWeight && markers && markers[pos] != null) {
          const marker = markers[pos];
          const markerOff = marker * H;
          const markerScale = markerScales ? markerScales[marker] : 1.0;
          val += markerWeight[markerOff + d] * markerScale;
        }
        s.x[destOff + d] = val;
      }
      if (charHashes) addCharEmbedding(s.x.subarray(destOff, destOff + H), charHashes[pos], charWeight, charScales, H);
    }

    // 2. Pre-LN Transformer blocks (Phase 3: every block, not just #0).
    // Each block reads and writes s.x in place, so stacking is just
    // iteration -- the scratch buffers are reused across blocks.
    for (let blk = 0; blk < this.numBlocks; blk++) {
      const B = `encoder.blocks.${blk}`;
      // LayerNorm 1 over x
      const ln1W = t[`${B}.ln1.weight`].data;
      const ln1B = t[`${B}.ln1.bias`].data;
      for (let pos = 0; pos < this.maxTokens; pos++) {
        const off = pos * H;
        for (let d = 0; d < H; d++) s.tokenBuf[d] = s.x[off + d];
        layerNorm(s.tokenBuf, ln1W, ln1B, s.tokenOut);
        for (let d = 0; d < H; d++) s.ln1_x[off + d] = s.tokenOut[d];
      }

      // Self-attention Q, K, V
      const qW = t[`${B}.attn.q_proj.weight`].data;
      const qS = t[`${B}.attn.q_proj.weight`].scales;
      const qB = t[`${B}.attn.q_proj.bias`].data;

      const kW = t[`${B}.attn.k_proj.weight`].data;
      const kS = t[`${B}.attn.k_proj.weight`].scales;
      const kB = t[`${B}.attn.k_proj.bias`].data;

      const vW = t[`${B}.attn.v_proj.weight`].data;
      const vS = t[`${B}.attn.v_proj.weight`].scales;
      const vB = t[`${B}.attn.v_proj.bias`].data;

      for (let pos = 0; pos < this.maxTokens; pos++) {
        const off = pos * H;
        for (let d = 0; d < H; d++) s.tokenBuf[d] = s.ln1_x[off + d];

        matvecLinear(s.tokenBuf, qW, qS, qB, s.tokenOut);
        for (let d = 0; d < H; d++) s.q[off + d] = s.tokenOut[d];

        matvecLinear(s.tokenBuf, kW, kS, kB, s.tokenOut);
        for (let d = 0; d < H; d++) s.k[off + d] = s.tokenOut[d];

        matvecLinear(s.tokenBuf, vW, vS, vB, s.tokenOut);
        for (let d = 0; d < H; d++) s.v[off + d] = s.tokenOut[d];
      }

      // Scaled Dot-Product Attention per Head
      const numHeads = this.numHeads;
      const headDim = this.headDim;
      const scaleFactor = 1.0 / Math.sqrt(headDim);

      s.attnOut.fill(0);

      for (let h = 0; h < numHeads; h++) {
        const hOff = h * headDim;
        for (let i = 0; i < this.maxTokens; i++) {
          if (mask[i] === 0) continue;
          const qOff = i * H + hOff;

          // Compute scores against all j
          let maxScore = -Infinity;
          const rowScores = new Float32Array(this.maxTokens);

          for (let j = 0; j < this.maxTokens; j++) {
            if (mask[j] === 0) {
              rowScores[j] = -10000.0;
              continue;
            }
            const kOff = j * H + hOff;
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += s.q[qOff + d] * s.k[kOff + d];
            }
            const sc = dot * scaleFactor;
            rowScores[j] = sc;
            if (sc > maxScore) maxScore = sc;
          }

          // Softmax
          let expSum = 0;
          for (let j = 0; j < this.maxTokens; j++) {
            if (mask[j] === 0) {
              rowScores[j] = 0;
            } else {
              const expVal = Math.exp(rowScores[j] - maxScore);
              rowScores[j] = expVal;
              expSum += expVal;
            }
          }
          const invExpSum = expSum > 0 ? 1.0 / expSum : 0;
          for (let j = 0; j < this.maxTokens; j++) rowScores[j] *= invExpSum;

          // Weighted sum of V
          const outDest = i * H + hOff;
          for (let j = 0; j < this.maxTokens; j++) {
            if (mask[j] === 0) continue;
            const vOff = j * H + hOff;
            const weight = rowScores[j];
            for (let d = 0; d < headDim; d++) {
              s.attnOut[outDest + d] += weight * s.v[vOff + d];
            }
          }
        }
      }

      // Output projection + residual 1 (a = x + out_proj(attnOut))
      const outProjW = t[`${B}.attn.out_proj.weight`].data;
      const outProjS = t[`${B}.attn.out_proj.weight`].scales;
      const outProjB = t[`${B}.attn.out_proj.bias`].data;

      for (let pos = 0; pos < this.maxTokens; pos++) {
        const off = pos * H;
        for (let d = 0; d < H; d++) s.tokenBuf[d] = s.attnOut[off + d];
        matvecLinear(s.tokenBuf, outProjW, outProjS, outProjB, s.tokenOut);
        for (let d = 0; d < H; d++) {
          s.x[off + d] += s.tokenOut[d]; // residual
        }
      }

      // LayerNorm 2
      const ln2W = t[`${B}.ln2.weight`].data;
      const ln2B = t[`${B}.ln2.bias`].data;
      for (let pos = 0; pos < this.maxTokens; pos++) {
        const off = pos * H;
        for (let d = 0; d < H; d++) s.tokenBuf[d] = s.x[off + d];
        layerNorm(s.tokenBuf, ln2W, ln2B, s.tokenOut);
        for (let d = 0; d < H; d++) s.ln2_a[off + d] = s.tokenOut[d];
      }

      // FFN + residual 2 (h = a + ffn2(gelu(ffn1(ln2_a))))
      const ffn1W = t[`${B}.ffn.0.weight`].data;
      const ffn1S = t[`${B}.ffn.0.weight`].scales;
      const ffn1B = t[`${B}.ffn.0.bias`].data;

      const ffn2W = t[`${B}.ffn.2.weight`].data;
      const ffn2S = t[`${B}.ffn.2.weight`].scales;
      const ffn2B = t[`${B}.ffn.2.bias`].data;

      for (let pos = 0; pos < this.maxTokens; pos++) {
        const off = pos * H;
        for (let d = 0; d < H; d++) s.tokenBuf[d] = s.ln2_a[off + d];

        matvecLinear(s.tokenBuf, ffn1W, ffn1S, ffn1B, s.ffn1);
        for (let f = 0; f < this.ffnDim; f++) s.ffn1Gelu[f] = gelu(s.ffn1[f]);

        matvecLinear(s.ffn1Gelu, ffn2W, ffn2S, ffn2B, s.ffn2);
        for (let d = 0; d < H; d++) {
          s.x[off + d] += s.ffn2[d]; // residual
        }
      }

    }

    // Final LayerNorm
    const finalLnW = t['encoder.final_ln.weight'].data;
    const finalLnB = t['encoder.final_ln.bias'].data;
    for (let pos = 0; pos < this.maxTokens; pos++) {
      const off = pos * H;
      for (let d = 0; d < H; d++) s.tokenBuf[d] = s.x[off + d];
      layerNorm(s.tokenBuf, finalLnW, finalLnB, s.tokenOut);
      for (let d = 0; d < H; d++) s.finalH[off + d] = s.tokenOut[d];
    }

    // 3. Context vector at targetPosition
    const targetOff = targetPosition * H;
    const contextVec = s.finalH.subarray(targetOff, targetOff + H);

    // 4. Option Scoring
    const numOptions = optionWordIds.length;
    const logits = new Float32Array(numOptions);

    const cpW = t['model.classical_proj.weight'].data;
    const cpS = t['model.classical_proj.weight'].scales;
    const cpB = t['model.classical_proj.bias'].data;

    const sc1W = t['model.option_scorer.0.weight'].data;
    const sc1S = t['model.option_scorer.0.weight'].scales;
    const sc1B = t['model.option_scorer.0.bias'].data;

    const sc2W = t['model.option_scorer.2.weight'].data;
    const sc2S = t['model.option_scorer.2.weight'].scales;
    const sc2B = t['model.option_scorer.2.bias'].data;

    const optTypeW = t['model.option_type_embedding.weight']?.data;
    const optTypeS = t['model.option_type_embedding.weight']?.scales;

    for (let opt = 0; opt < numOptions; opt++) {
      if (optionMask && optionMask[opt] === 0) {
        logits[opt] = -10000.0;
        continue;
      }

      const optWord = optionWordIds[opt];
      const optRowOff = optWord * H;
      const optScale = wordScales[optWord];

      // Option embedding
      const optEmb = new Float32Array(H);
      for (let d = 0; d < H; d++) optEmb[d] = wordWeight[optRowOff + d] * optScale;
      addCharEmbedding(optEmb, optionCharHashes?.[opt], charWeight, charScales, H);

      // Classical features projection
      const rawFeat = new Float32Array(classicalFeatures[opt]);
      matvecLinear(rawFeat, cpW, cpS, cpB, s.classProj);

      // Construct z = [contextVec, optEmb, contextVec * optEmb, classProj, optTypeEmb]
      let zIdx = 0;
      for (let d = 0; d < H; d++) s.z[zIdx++] = contextVec[d];
      for (let d = 0; d < H; d++) s.z[zIdx++] = optEmb[d];
      for (let d = 0; d < H; d++) s.z[zIdx++] = contextVec[d] * optEmb[d];
      for (let d = 0; d < 16; d++) s.z[zIdx++] = s.classProj[d];

      const optType = opt === 0 ? 0 : 1;
      if (optTypeW) {
        const typeOff = optType * 4;
        const typeScale = optTypeS ? optTypeS[optType] : 1.0;
        for (let d = 0; d < 4; d++) s.z[zIdx++] = optTypeW[typeOff + d] * typeScale;
      } else {
        for (let d = 0; d < 4; d++) s.z[zIdx++] = 0;
      }

      // Option Scorer Layer 1
      matvecLinear(s.z, sc1W, sc1S, sc1B, s.sc1);
      for (let d = 0; d < H; d++) s.sc1Gelu[d] = gelu(s.sc1[d]);

      // Option Scorer Layer 2 (scalar logit)
      matvecLinear(s.sc1Gelu, sc2W, sc2S, sc2B, s.sc2);
      logits[opt] = s.sc2[0];
    }

    // 5. Softmax over options
    let maxLogit = -Infinity;
    for (let opt = 0; opt < numOptions; opt++) {
      if (optionMask && optionMask[opt] === 0) continue;
      if (logits[opt] > maxLogit) maxLogit = logits[opt];
    }

    const probs = new Float32Array(numOptions);
    let expSum = 0;
    for (let opt = 0; opt < numOptions; opt++) {
      if (optionMask && optionMask[opt] === 0) {
        probs[opt] = 0.0;
      } else {
        const expVal = Math.exp(logits[opt] - maxLogit);
        probs[opt] = expVal;
        expSum += expVal;
      }
    }
    const invExpSum = expSum > 0 ? 1.0 / expSum : 0;
    for (let opt = 0; opt < numOptions; opt++) {
      probs[opt] *= invExpSum;
    }

    // Selected choice
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let opt = 0; opt < numOptions; opt++) {
      if (optionMask && optionMask[opt] === 0) continue;
      if (logits[opt] > bestScore) {
        bestScore = logits[opt];
        bestIdx = opt;
      }
    }

    return {
      logits: Array.from(logits),
      probabilities: Array.from(probs),
      selectedIndex: bestIdx,
    };
  }
}

export function loadAttentionReranker({ binPath, metaPath }) {
  if (!existsSync(binPath)) {
    throw new Error(`Attention reranker binary artifact not found: ${binPath}`);
  }
  if (!existsSync(metaPath)) {
    throw new Error(`Attention reranker metadata not found: ${metaPath}`);
  }

  const rawMeta = readFileSync(metaPath, 'utf8');
  const metadata = JSON.parse(rawMeta);
  if (!metadata || (metadata.schema !== 'attention-reranker-artifact-v1'
    && metadata.schema !== 'attention-reranker-v1')) {
    throw new Error('Invalid attention reranker metadata schema');
  }
  if (!Number.isInteger(metadata.k) || ![4, 6, 8].includes(metadata.k)) {
    throw new Error('Invalid attention reranker shortlist K');
  }
  if (metadata.tokenizerVersion != null
    && metadata.tokenizerVersion !== 'attention-tokenizer-v1') {
    throw new Error('Incompatible attention tokenizer version');
  }

  const binBuffer = readFileSync(binPath);
  if (metadata.binSize != null && metadata.binSize !== binBuffer.byteLength) {
    throw new Error('Attention reranker binary size mismatch');
  }
  if (metadata.binHash) {
    const gotHash = createHash('sha256').update(binBuffer).digest('hex');
    if (gotHash !== metadata.binHash) {
      throw new Error('Attention reranker binary hash mismatch');
    }
  }
  const dataView = new DataView(binBuffer.buffer, binBuffer.byteOffset, binBuffer.byteLength);

  // Validate Header (32 bytes)
  const magic = binBuffer.subarray(0, 8).toString('ascii');
  if (magic !== MAGIC_STR) {
    throw new Error(`Invalid attention reranker magic: "${magic}", expected "${MAGIC_STR}"`);
  }
  const formatVer = dataView.getUint32(8, true);
  if (formatVer !== FORMAT_VERSION) {
    throw new Error(`Unsupported attention reranker format version: ${formatVer}`);
  }
  const tensorCount = dataView.getUint32(12, true);
  const tableOffset = dataView.getUint32(16, true);

  const tensors = {};

  // Parse tensor records
  for (const [name, info] of Object.entries(metadata.tensors)) {
    const dOff = info.dataOffset;
    const dLen = info.dataLength;
    const sOff = info.scaleOffset;
    const sLen = info.scaleLength;
    const dtype = info.dtype;
    const dims = info.dims;
    if (!Number.isInteger(dOff) || !Number.isInteger(dLen) || dOff < 0 || dLen < 0
      || dOff + dLen > binBuffer.byteLength) {
      throw new Error(`Attention reranker tensor ${name} data bounds invalid`);
    }
    if (!Array.isArray(dims) || dims.some((dim) => !Number.isInteger(dim) || dim < 0)) {
      throw new Error(`Attention reranker tensor ${name} shape invalid`);
    }
    if (!Number.isInteger(sOff) || !Number.isInteger(sLen) || sOff < 0 || sLen < 0
      || sOff + sLen > binBuffer.byteLength) {
      throw new Error(`Attention reranker tensor ${name} scale bounds invalid`);
    }
    const elementBytes = dtype === 'int8' ? 1 : 4;
    if (dOff % elementBytes !== 0 || sOff % 4 !== 0 || (sLen % 4) !== 0) {
      throw new Error(`Attention reranker tensor ${name} alignment invalid`);
    }

    if (dtype === 'int8') {
      const dataArr = new Int8Array(binBuffer.buffer, binBuffer.byteOffset + dOff, dLen);
      let scalesArr = null;
      if (sLen > 0) {
        scalesArr = new Float32Array(binBuffer.buffer, binBuffer.byteOffset + sOff, sLen / 4);
      }
      tensors[name] = {
        name,
        dtype,
        dims,
        data: dataArr,
        scales: scalesArr,
      };
    } else {
      const dataArr = new Float32Array(binBuffer.buffer, binBuffer.byteOffset + dOff, dLen / 4);
      tensors[name] = {
        name,
        dtype,
        dims,
        data: dataArr,
        scales: null,
      };
    }
  }

  return new AttentionRerankerInstance(metadata, tensors, binBuffer);
}

export function scoreRankingOptions(reranker, args) {
  return reranker.scoreOptions(args);
}
