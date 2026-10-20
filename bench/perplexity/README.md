# bench/perplexity — held-out texts for the P4 perplexity metric

Two small plain-text corpora, neither drawn from the stage-4 training data
(Magicoder-OSS-Instruct-75K, first 20 000 records scanned, 3000 drawn).  They
exist so `scripts/bench_finetune.sh` can measure perplexity per model on text the
fine-tune never saw — the classic check that a LoRA did not degrade the base.

| file | source | licence |
|---|---|---|
| `general.txt` | Wikipedia, "Containerization", revision 1376287797 (2026-09-23), plain-text extract via the MediaWiki API (`action=query&prop=extracts&explaintext=1&revids=1376287797`) | CC BY-SA 4.0 |
| `code.txt` | CPython v3.12.0, `Lib/heapq.py` (blob 2fd9d1ff4bf827b4c919253be98a7ee0c3f7b829) concatenated with `Lib/textwrap.py` (blob 98bedd27ea3a11995e2bf81a115df999767dc134) | PSF-2.0 |

Both stay under ~50 KB.  `code.txt` is the concatenation of two stdlib modules
because one alone tokenizes to fewer than 2 x ctx tokens, which is below the
minimum `llama perplexity` needs at `-c 4096`; two raise it comfortably above the
threshold while staying inside the size budget.  The header comment inside each
file repeats the exact revision/blob, so the corpora are reproducible without
re-deriving them from a live source.
