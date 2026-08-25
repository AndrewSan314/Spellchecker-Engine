// Download and Ingest script for Vietnamese Datasets
// 1. Viwiki-Spelling (heraclex12/Viwiki-spelling)
// 2. Underthesea Vietnamese Wikipedia (undertheseanlp/corpus.viwiki)
// 3. VSEC spelling dataset / benchmarks
// 4. BinhVQ News Corpus sample extracts

import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RAW_DATA_DIR = path.join(HERE, '..', 'dataset_raw');

if (!existsSync(RAW_DATA_DIR)) {
  mkdirSync(RAW_DATA_DIR, { recursive: true });
}

async function fetchJsonOrText(url, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) {
      return { ok: false, status: res.status, statusText: res.statusText };
    }
    const text = await res.text();
    return { ok: true, data: text };
  } catch (err) {
    clearTimeout(timer);
    return { ok: false, error: err.message };
  }
}

async function main() {
  console.log('═══════ DOWNLOADING VIETNAMESE DATASETS ═══════\n');

  // 1. Probe heraclex12/Viwiki-spelling
  console.log('[1/4] Probing Viwiki-Spelling GitHub repository...');
  const viwikiCandidateUrls = [
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/master/data/test.jsonl',
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/main/data/test.jsonl',
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/master/test.jsonl',
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/main/test.jsonl',
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/master/data.jsonl',
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/main/data.jsonl',
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/master/README.md',
    'https://api.github.com/repos/heraclex12/Viwiki-spelling/contents'
  ];

  for (const url of viwikiCandidateUrls) {
    const res = await fetchJsonOrText(url);
    if (res.ok) {
      console.log(`  ✔ Found: ${url} (${res.data.length} bytes)`);
      if (url.endsWith('.jsonl') || url.includes('/contents') || url.endsWith('.md')) {
        const fname = url.includes('/contents') ? 'viwiki_contents.json' : path.basename(url);
        writeFileSync(path.join(RAW_DATA_DIR, `viwiki_${fname}`), res.data, 'utf8');
      }
    } else {
      console.log(`  ✘ ${url} -> ${res.status ?? res.error}`);
    }
  }

  // 2. Probe undertheseanlp/corpus.viwiki
  console.log('\n[2/4] Probing Underthesea Wikipedia repository...');
  const undertheseaUrls = [
    'https://api.github.com/repos/undertheseanlp/corpus.viwiki/contents',
    'https://raw.githubusercontent.com/undertheseanlp/corpus.viwiki/master/README.md',
    'https://raw.githubusercontent.com/undertheseanlp/corpus.viwiki/main/README.md',
  ];

  for (const url of undertheseaUrls) {
    const res = await fetchJsonOrText(url);
    if (res.ok) {
      console.log(`  ✔ Found: ${url} (${res.data.length} bytes)`);
      const fname = url.includes('/contents') ? 'underthesea_contents.json' : 'underthesea_readme.md';
      writeFileSync(path.join(RAW_DATA_DIR, fname), res.data, 'utf8');
    } else {
      console.log(`  ✘ ${url} -> ${res.status ?? res.error}`);
    }
  }

  // 3. Probe Hugging Face / BinhVQ / VSEC repositories
  console.log('\n[3/4] Probing Hugging Face Vietnamese corpora...');
  const hfUrls = [
    'https://huggingface.co/datasets/undertheseanlp/corpus.viwiki/raw/main/README.md',
    'https://huggingface.co/api/datasets/undertheseanlp/corpus.viwiki',
    'https://huggingface.co/api/datasets/binhvq/news-corpus',
  ];

  for (const url of hfUrls) {
    const res = await fetchJsonOrText(url);
    if (res.ok) {
      console.log(`  ✔ Found: ${url} (${res.data.length} bytes)`);
    } else {
      console.log(`  ✘ ${url} -> ${res.status ?? res.error}`);
    }
  }
}

main().catch(err => console.error('Download script error:', err));
