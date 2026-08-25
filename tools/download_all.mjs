// Comprehensive Downloader for Vietnamese NLP Benchmarks & Corpora
import { writeFileSync, mkdirSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RAW_DATA_DIR = path.join(HERE, '..', 'dataset_raw');

if (!existsSync(RAW_DATA_DIR)) {
  mkdirSync(RAW_DATA_DIR, { recursive: true });
}

async function downloadFile(url, targetFilename, isJson = false) {
  console.log(`Downloading: ${url} -> ${targetFilename}...`);
  try {
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
    });
    if (!res.ok) {
      console.log(`  ✘ HTTP ${res.status}: ${res.statusText}`);
      return false;
    }
    const buffer = Buffer.from(await res.arrayBuffer());
    const outPath = path.join(RAW_DATA_DIR, targetFilename);
    writeFileSync(outPath, buffer);
    console.log(`  ✔ Saved ${buffer.length.toLocaleString()} bytes to ${outPath}`);
    return true;
  } catch (err) {
    console.log(`  ✘ Error: ${err.message}`);
    return false;
  }
}

async function main() {
  console.log('════════ DOWNLOADING STANDARD VIETNAMESE DATASETS ════════\n');

  // 1. Viwiki-Spelling Dataset (107 documents with 1500+ labeled spelling mistakes)
  console.log('--- [1/3] Viwiki-Spelling Dataset (heraclex12/Viwiki-spelling) ---');
  await downloadFile(
    'https://raw.githubusercontent.com/heraclex12/Viwiki-spelling/main/spelling_test.json',
    'viwiki_spelling_test.json'
  );

  // 2. Underthesea Vietnamese Wikipedia Corpus
  console.log('\n--- [2/3] Underthesea Vietnamese Wikipedia Corpus ---');
  await downloadFile(
    'https://raw.githubusercontent.com/undertheseanlp/corpus.viwiki/master/crawled-pages.txt',
    'viwiki_crawled_pages.txt'
  );

  // Fetch article files from Underthesea
  try {
    const treeRes = await fetch('https://api.github.com/repos/undertheseanlp/corpus.viwiki/contents/viwiki?ref=master', {
      headers: { 'User-Agent': 'Mozilla/5.0' },
    });
    if (treeRes.ok) {
      const items = await treeRes.json();
      console.log(`Found ${items.length} files in undertheseanlp/corpus.viwiki/viwiki`);
      for (const item of items.slice(0, 10)) {
        if (item.download_url) {
          await downloadFile(item.download_url, `underthesea_${item.name}`);
        }
      }
    }
  } catch (e) {
    console.log(`  Could not list underthesea directory: ${e.message}`);
  }

  // 3. VSEC spelling dataset sources
  console.log('\n--- [3/3] VSEC Vietnamese Spelling Correction Dataset ---');
  const vsecUrls = [
    'https://raw.githubusercontent.com/halannhile/vietnamese-spelling-error-detection/main/data/test.json',
    'https://raw.githubusercontent.com/halannhile/vietnamese-spelling-error-detection/main/data/train.json',
    'https://raw.githubusercontent.com/halannhile/vietnamese-spelling-error-detection/main/README.md',
  ];

  for (const u of vsecUrls) {
    const fname = `vsec_${path.basename(u)}`;
    await downloadFile(u, fname);
  }

  console.log('\n✅ Download phase complete. Checking files in dataset_raw/ ...');
}

main().catch(err => console.error(err));
