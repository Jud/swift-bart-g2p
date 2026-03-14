# autoresearch: G2P Neural Reranker

Autonomous research to improve the neural reranker for BART G2P beam rescoring.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `Sources/BARTG2P/NeuralReranker.swift` — Swift inference struct. Must match Python model.
   - `Sources/BARTG2P/BARTG2P.swift` — Integration: `predictWithLMRescore()` scoring logic.
   - `scripts/train_reranker.py` — PyTorch training script. Primary file you modify.
   - `Tests/BARTG2PTests/BARTG2PTests.swift` — Accuracy tests and thresholds.
4. **Verify training data exists**: Check `data/reranker_train.tsv` (~940K rows, beam=8).
   If missing: `swift run -c release GenerateRerankerData data_cmudict_full_ipa.tsv data/reranker_train.tsv`
5. **Verify venv**: `.venv/bin/python3 -c "import torch; print(torch.__version__)"` should work.
   If not: `python3 -m venv .venv && .venv/bin/pip install torch numpy`
6. **Initialize results.tsv**: Create with just the header row.
7. **Confirm and go**.

## Experimentation

### What you CAN modify
- `scripts/train_reranker.py` — architecture, loss function, hyperparameters, features
- `Sources/BARTG2P/NeuralReranker.swift` — inference code (must match Python model)
- `Sources/BARTG2P/BARTG2P.swift` — scoring logic in `predictWithLMRescore()`
- Beam width, scoring combination, candidate filtering

### What you CANNOT modify
- `Sources/BARTG2P/BARTG2P.swift` — anything outside `predictWithLMRescore()` and the reranker property
- `Sources/GenerateRerankerData/main.swift` — data generator (pre-generated data is fixed)
- `prepare.py` equivalent: the BART model weights, trigram LM, test infrastructure
- Test evaluation logic (normalize, phonemeErrorRate, measureDictAccuracy)

### The goal: maximize loose accuracy on the 1000-word CMUdict test set.

Current baseline: **56.4% loose accuracy** (beam=8, 36K-param 1D CNN reranker).
Oracle ceiling beam=8: 75.0%. Oracle beam=16: 80.4%.

### Constraints
- **Model size**: Keep safetensors under ~500KB. Current: 142KB (36K params).
- **Inference speed**: Keep rescoreLM under ~10ms/word. Current: 3.4ms.
- **Simplicity**: All else equal, simpler is better. Don't add complexity for <0.5% gain.
- **Swift parity**: If you change the Python model, update NeuralReranker.swift to match.

## The experiment loop

**Turn time budget: 40 minutes** (~30 min training + 10 min overhead).

LOOP FOREVER:

1. Look at git state and results.tsv for context
2. Modify in-scope files with an experimental idea
3. `git commit` (message: short description of what you're trying)
4. Train: `.venv/bin/python3 -u scripts/train_reranker.py data/reranker_train.tsv Sources/BARTG2P/Resources/reranker.safetensors --epochs 30 --batch-size 64 --patience 5 > run.log 2>&1`
5. Read results: `grep "val_acc\|Test acc\|Exported" run.log | tail -3`
6. If training crashed, check `tail -n 50 run.log`, attempt fix (max 2 retries)
7. If Swift inference changed, rebuild: `swift build -c release 2>&1 | grep error`
8. Test: `swift test -c release --filter dictAccuracyRescoreLM 2>&1 | grep "dict_accuracy"`
9. Record in results.tsv and update FINDINGS.md
10. If loose accuracy improved: keep the commit, advance the branch
11. If equal or worse: `git reset --hard HEAD~1` to discard

### Output format

Training prints per-epoch val_acc. Key metrics to extract:
```
grep "Best val_acc\|Test acc" run.log
```

Swift test prints:
```
dict_accuracy_rescoreLM: exact=347/1000 (34.7%) loose=564/1000 (56.4%) PER=11.2%
```

## Logging results

Log to `results.tsv` (tab-separated). Do NOT commit this file.

```
commit	loose	exact	val_acc	per	status	description
```

Example:
```
commit	loose	exact	val_acc	per	status	description
a1b2c3d	564	347	0.7461	11.2	keep	baseline (ListMLE beam=8 36K params)
b2c3d4e	580	355	0.7612	10.8	keep	add 3rd conv layer
c3d4e5f	550	340	0.7300	11.5	discard	cross-entropy loss instead of ListMLE
```

After each experiment, also update FINDINGS.md with a summary of what was tried and learned.

## Research directions to explore

- **Architecture**: wider channels (64→96), 3rd conv layer, residual connections, attention pooling instead of avg pool
- **Loss function**: margin-based ranking, contrastive loss, label smoothing
- **Features**: per-token logprob (modelLP/len), candidate length, phoneme-grapheme length ratio
- **Training**: curriculum learning, hard negative mining, data augmentation
- **Scoring**: hybrid reranker+trigram combination, temperature scaling
- **Beam strategy**: beam=16 with top-8 filtering for reranker

## Key context

- Oracle beam=8 is 75% — 25% of test words simply don't have a correct candidate
- Ranking accuracy ~74.5% → effective accuracy = 0.75 * 0.745 ≈ 55.9% (we get 56.4%)
- The model is near theoretical max for beam=8 + 36K params
- Beam=16 oracle is 80.4% but ranking accuracy drops to 67% (net wash)
- Feature normalization and dropout both hurt in experiments so far

**NEVER STOP**: Once the loop begins, do not pause to ask the human. Run experiments indefinitely until manually interrupted.
