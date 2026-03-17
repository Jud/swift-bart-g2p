# swift-bart-g2p

neural grapheme-to-phoneme in pure Swift. feed it a word, get IPA phonemes back.

BART-tiny architecture. 752K parameters. 3MB weights. only dependency is Accelerate.

built for OOV word pronunciation -- when your TTS engine hits "kubernetes" and has no idea what to do.

## install

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/Jud/swift-bart-g2p.git", branch: "main"),
]
```

## usage

```swift
import BARTG2P

let g2p = BARTG2P.fromBundle()!

g2p.predict("kubernetes")                     // "kˌubəɹnˈits" (~1.7ms, beam + reranker)
g2p.predict("kubernetes", rescoreLM: false)   // greedy only (~0.6ms, slightly less accurate)
```

## architecture

```mermaid
graph LR
    A[input word] --> B[grapheme tokenizer]
    B --> C[BART encoder<br/>self-attn + FFN]
    C --> D[BART decoder<br/>causal self-attn +<br/>cross-attn + FFN]
    D --> E[diverse beam search]
    E --> F[MBR + reranker]
    F --> G[IPA phonemes]
```

single encoder layer, single decoder layer. KV-cached autoregressive decoding -- diverse beam search generates candidates, MBR consensus and a neural reranker pick the winner. all matrix ops go through `cblas_sgemm` / `vDSP`. weights loaded from safetensors at init.

## accuracy

on 1000-word CMUdict sample:

| mode | exact match | loose match | PER |
|------|-------------|-------------|-----|
| greedy | 35.7% | 49.5% | 12.8% |
| rescoreLM (default) | 35.5% | 56.3% | 11.2% |

"loose" normalizes stress markers and allophones before comparison. PER = phoneme error rate.

## hard words

the stuff dictionaries can't help you with.

| word | output | what's hard |
|------|--------|-------------|
| debt | dˈɛt | silent b |
| receipt | ɹəsˈit | silent p |
| yacht | jˈɑt | silent ch |
| bourgeois | bʊɹʒwˈɑ | french loanword |
| espresso | ɛspɹˈɛsO | not "expresso" |
| kubernetes | kˌubəɹnˈits | tech term, pure OOV |
| webpack | wˈɛbpˌæk | compound neologism |
| onomatopoeia | ˌɑnəmˌæɾəpˈiə | 6 syllables, greek roots |
| antidisestablishmentarianism | ˌæntɪdˌɪsəstˈæblɪʃməntˈɛɹiənˌɪzəm | 12 syllables, stress everywhere |

## model

- **source**: [PeterReid/graphemes_to_phonemes_en_us](https://huggingface.co/PeterReid/graphemes_to_phonemes_en_us)
- **d_model**: 128
- **layers**: 1 encoder, 1 decoder
- **attention heads**: 1 per layer
- **FFN dim**: 1024
- **vocab**: 63 tokens (graphemes + phonemes, shared embedding)
- **weights**: safetensors, ~3MB bundled in SPM resources
- **platform**: macOS 14+

## license

Apache 2.0
