import Accelerate
import Foundation

/// BART-tiny neural G2P for out-of-vocabulary words.
///
/// 752K-param BART (d_model=128, 1 encoder layer, 1 decoder layer, vocab 63).
/// Weights from `PeterReid/graphemes_to_phonemes_en_us` (3 MB safetensors).
/// Pure Accelerate implementation — zero external dependencies.
public final class BARTG2P {
    private static let d = 128
    private static let ffnDim = 1024
    private static let vocabSize = 63
    private static let maxPos = 64
    private static let posOff = 2
    private static let bosId = 1
    private static let eosId = 2
    private static let scale: Float = 1.0 / sqrtf(Float(d))

    private let graphemeToId: [Character: Int]
    private let idToPhoneme: [Int: Character]

    /// Phoneme trigram LM for beam rescoring (lazy-loaded).
    private var trigramLM: PhonemeTrigramLM?

    /// Neural reranker for candidate rescoring (lazy-loaded).
    private var reranker: NeuralReranker?
    private var rerankerChecked = false

    // All weights stored as flat [Float] arrays.
    // Convention: "W" = weight matrix [outDim, inDim], "B" = bias [outDim].
    private let sharedW: [Float]       // [63, 128]
    private let logitBias: [Float]     // [63]

    // Encoder
    private let ePosW: [Float]         // [66, 128]
    private let eLnEW: [Float], eLnEB: [Float]
    private let eQW: [Float], eQB: [Float], eKW: [Float], eKB: [Float]
    private let eVW: [Float], eVB: [Float], eOW: [Float], eOB: [Float]
    private let eSaLnW: [Float], eSaLnB: [Float]
    private let eF1W: [Float], eF1B: [Float], eF2W: [Float], eF2B: [Float]
    private let eFLnW: [Float], eFLnB: [Float]

    // Decoder
    private let dPosW: [Float]         // [66, 128]
    private let dLnEW: [Float], dLnEB: [Float]
    private let dSQW: [Float], dSQB: [Float], dSKW: [Float], dSKB: [Float]
    private let dSVW: [Float], dSVB: [Float], dSOW: [Float], dSOB: [Float]
    private let dSaLnW: [Float], dSaLnB: [Float]
    private let dCQW: [Float], dCQB: [Float], dCKW: [Float], dCKB: [Float]
    private let dCVW: [Float], dCVB: [Float], dCOW: [Float], dCOB: [Float]
    private let dCaLnW: [Float], dCaLnB: [Float]
    private let dF1W: [Float], dF1B: [Float], dF2W: [Float], dF2B: [Float]
    private let dFLnW: [Float], dFLnB: [Float]

    public init?(weightsURL: URL, configURL: URL) {
        guard let cfgData = try? Data(contentsOf: configURL),
              let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
              let gChars = cfg["grapheme_chars"] as? String,
              let pChars = cfg["phoneme_chars"] as? String
        else { return nil }

        var g2id = [Character: Int]()
        for (i, c) in gChars.enumerated() where i >= 4 { g2id[c] = i }
        self.graphemeToId = g2id

        var id2p = [Int: Character]()
        for (i, c) in pChars.enumerated() where i >= 4 { id2p[i] = c }
        self.idToPhoneme = id2p

        guard let st = try? SafetensorsLoader(url: weightsURL) else { return nil }
        do {
            sharedW = try st.floats(for: "model.shared.weight")
            logitBias = try st.floats(for: "final_logits_bias")

            ePosW = try st.floats(for: "model.encoder.embed_positions.weight")
            eLnEW = try st.floats(for: "model.encoder.layernorm_embedding.weight")
            eLnEB = try st.floats(for: "model.encoder.layernorm_embedding.bias")
            eQW = try st.floats(for: "model.encoder.layers.0.self_attn.q_proj.weight")
            eQB = try st.floats(for: "model.encoder.layers.0.self_attn.q_proj.bias")
            eKW = try st.floats(for: "model.encoder.layers.0.self_attn.k_proj.weight")
            eKB = try st.floats(for: "model.encoder.layers.0.self_attn.k_proj.bias")
            eVW = try st.floats(for: "model.encoder.layers.0.self_attn.v_proj.weight")
            eVB = try st.floats(for: "model.encoder.layers.0.self_attn.v_proj.bias")
            eOW = try st.floats(for: "model.encoder.layers.0.self_attn.out_proj.weight")
            eOB = try st.floats(for: "model.encoder.layers.0.self_attn.out_proj.bias")
            eSaLnW = try st.floats(for: "model.encoder.layers.0.self_attn_layer_norm.weight")
            eSaLnB = try st.floats(for: "model.encoder.layers.0.self_attn_layer_norm.bias")
            eF1W = try st.floats(for: "model.encoder.layers.0.fc1.weight")
            eF1B = try st.floats(for: "model.encoder.layers.0.fc1.bias")
            eF2W = try st.floats(for: "model.encoder.layers.0.fc2.weight")
            eF2B = try st.floats(for: "model.encoder.layers.0.fc2.bias")
            eFLnW = try st.floats(for: "model.encoder.layers.0.final_layer_norm.weight")
            eFLnB = try st.floats(for: "model.encoder.layers.0.final_layer_norm.bias")

            dPosW = try st.floats(for: "model.decoder.embed_positions.weight")
            dLnEW = try st.floats(for: "model.decoder.layernorm_embedding.weight")
            dLnEB = try st.floats(for: "model.decoder.layernorm_embedding.bias")
            dSQW = try st.floats(for: "model.decoder.layers.0.self_attn.q_proj.weight")
            dSQB = try st.floats(for: "model.decoder.layers.0.self_attn.q_proj.bias")
            dSKW = try st.floats(for: "model.decoder.layers.0.self_attn.k_proj.weight")
            dSKB = try st.floats(for: "model.decoder.layers.0.self_attn.k_proj.bias")
            dSVW = try st.floats(for: "model.decoder.layers.0.self_attn.v_proj.weight")
            dSVB = try st.floats(for: "model.decoder.layers.0.self_attn.v_proj.bias")
            dSOW = try st.floats(for: "model.decoder.layers.0.self_attn.out_proj.weight")
            dSOB = try st.floats(for: "model.decoder.layers.0.self_attn.out_proj.bias")
            dSaLnW = try st.floats(for: "model.decoder.layers.0.self_attn_layer_norm.weight")
            dSaLnB = try st.floats(for: "model.decoder.layers.0.self_attn_layer_norm.bias")
            dCQW = try st.floats(for: "model.decoder.layers.0.encoder_attn.q_proj.weight")
            dCQB = try st.floats(for: "model.decoder.layers.0.encoder_attn.q_proj.bias")
            dCKW = try st.floats(for: "model.decoder.layers.0.encoder_attn.k_proj.weight")
            dCKB = try st.floats(for: "model.decoder.layers.0.encoder_attn.k_proj.bias")
            dCVW = try st.floats(for: "model.decoder.layers.0.encoder_attn.v_proj.weight")
            dCVB = try st.floats(for: "model.decoder.layers.0.encoder_attn.v_proj.bias")
            dCOW = try st.floats(for: "model.decoder.layers.0.encoder_attn.out_proj.weight")
            dCOB = try st.floats(for: "model.decoder.layers.0.encoder_attn.out_proj.bias")
            dCaLnW = try st.floats(for: "model.decoder.layers.0.encoder_attn_layer_norm.weight")
            dCaLnB = try st.floats(for: "model.decoder.layers.0.encoder_attn_layer_norm.bias")
            dF1W = try st.floats(for: "model.decoder.layers.0.fc1.weight")
            dF1B = try st.floats(for: "model.decoder.layers.0.fc1.bias")
            dF2W = try st.floats(for: "model.decoder.layers.0.fc2.weight")
            dF2B = try st.floats(for: "model.decoder.layers.0.fc2.bias")
            dFLnW = try st.floats(for: "model.decoder.layers.0.final_layer_norm.weight")
            dFLnB = try st.floats(for: "model.decoder.layers.0.final_layer_norm.bias")
        } catch {
            return nil
        }
    }

    /// Load from the module's bundled resources.
    public static func fromBundle() -> BARTG2P? {
        guard let wURL = Bundle.module.url(forResource: "bart_g2p", withExtension: "safetensors"),
              let cURL = Bundle.module.url(forResource: "bart_g2p_config", withExtension: "json")
        else { return nil }
        return BARTG2P(weightsURL: wURL, configURL: cURL)
    }

    /// Predict IPA phonemes for a lowercased word.
    /// - Parameters:
    ///   - word: Input word (lowercased).
    ///   - beamWidth: Number of beams. 1 = greedy (default), >1 = beam search.
    ///   - lengthPenalty: Exponent for length normalization in beam search (0 = none, 1 = full).
    ///   - rescoreLM: When true, uses beam=8 and rescores with a phoneme trigram LM for better accuracy.
    public func predict(_ word: String, beamWidth: Int = 1, lengthPenalty: Float = 0.0,
                        rescoreLM: Bool = true) -> String? {
        if rescoreLM {
            return predictWithLMRescore(word)
        }

        let gIds = word.compactMap { graphemeToId[$0] }
        guard !gIds.isEmpty, gIds.count + 2 <= Self.maxPos else { return nil }

        let encIn = [Self.bosId] + gIds + [Self.eosId]
        let encOut = encode(encIn)
        let d = Self.d
        let cK = linear(encOut, rows: encIn.count, wt: dCKW, b: dCKB, outD: d)
        let cV = linear(encOut, rows: encIn.count, wt: dCVW, b: dCVB, outD: d)
        let pIds: [Int]
        if beamWidth <= 1 {
            pIds = decode(cK: cK, cV: cV, encLen: encIn.count)
        } else {
            guard let best = beamSearchCore(cK: cK, cV: cV, encLen: encIn.count,
                                            beamWidth: beamWidth, lengthPenalty: lengthPenalty).first
            else { return nil }
            pIds = Array(best.tokens.dropFirst().filter { $0 >= 4 && $0 != Self.eosId })
        }
        guard !pIds.isEmpty else { return nil }

        let chars = pIds.compactMap { idToPhoneme[$0] }
        return chars.isEmpty ? nil : String(chars)
    }

    /// Return trigram LM log probability for a phoneme string.
    public func trigramScore(_ ipa: String) -> Float {
        if trigramLM == nil {
            if let url = Bundle.module.url(forResource: "phoneme_trigram", withExtension: "tsv") {
                trigramLM = PhonemeTrigramLM(url: url)
            }
        }
        return trigramLM?.logProb(ipa) ?? 0
    }

    /// Predict using diverse beam search + MBR consensus + neural reranker (or trigram LM fallback).
    private func predictWithLMRescore(_ word: String) -> String? {
        // Lazy-load trigram LM
        if trigramLM == nil {
            if let url = Bundle.module.url(forResource: "phoneme_trigram", withExtension: "tsv") {
                trigramLM = PhonemeTrigramLM(url: url)
            }
        }

        // Lazy-load neural reranker (one attempt)
        if !rerankerChecked {
            rerankerChecked = true
            if let url = Bundle.module.url(forResource: "reranker", withExtension: "safetensors") {
                reranker = NeuralReranker(url: url)
            }
        }

        let beamW = reranker != nil ? 8 : 4
        let candidates = predictNBestScored(word, beamWidth: beamW, diversityPenalty: 1.5)
        guard !candidates.isEmpty else { return nil }
        guard let lm = trigramLM else { return candidates[0].text }

        // Compute MBR consensus scores (probability-weighted avg edit distance)
        let n = candidates.count
        let mbrScores: [Float]
        if n > 1 {
            let maxLP = candidates.map(\.logProb).max()!
            let rawProbs = candidates.map { expf($0.logProb - maxLP) }
            let probSum = rawProbs.reduce(0, +)
            let probs = rawProbs.map { $0 / probSum }

            var scores = [Float](repeating: 0, count: n)
            for i in 0..<n {
                for j in (i+1)..<n {
                    let d = Float(Self.phonemeEditDistance(candidates[i].text, candidates[j].text))
                    scores[i] += probs[j] * d
                    scores[j] += probs[i] * d
                }
            }
            mbrScores = scores
        } else {
            mbrScores = [0]
        }

        let mbrWeight: Float = 0.3

        // Neural reranker path + MBR
        if let rr = reranker {
            var bestScore: Float = -.infinity
            var bestText: String = candidates[0].text
            for (i, (text, modelLP)) in candidates.enumerated() {
                let trigramLP = lm.logProb(text)
                let rrScore = rr.score(word: word, candidate: text,
                                       modelLogProb: modelLP, trigramLogProb: trigramLP)
                let score = rrScore - mbrWeight * mbrScores[i]
                if score > bestScore {
                    bestScore = score
                    bestText = text
                }
            }
            return bestText
        }

        // Fallback: trigram + MBR rescoring
        let lambda: Float = 0.35
        var bestScore: Float = -.infinity
        var bestText: String = candidates[0].text
        for (i, (text, modelLP)) in candidates.enumerated() {
            let combined = modelLP + lambda * lm.logProb(text) - mbrWeight * mbrScores[i]
            if combined > bestScore {
                bestScore = combined
                bestText = text
            }
        }
        return bestText
    }

    /// Return all beam candidates sorted by score (best first).
    public func predictNBest(_ word: String, beamWidth: Int, lengthPenalty: Float = 0.0,
                             diversityPenalty: Float = 0.0) -> [String] {
        return predictNBestScored(word, beamWidth: beamWidth, lengthPenalty: lengthPenalty,
                                  diversityPenalty: diversityPenalty)
            .map { $0.text }
    }

    /// Return all beam candidates with their log-probabilities, sorted best-first.
    public func predictNBestScored(_ word: String, beamWidth: Int, lengthPenalty: Float = 0.0,
                                   diversityPenalty: Float = 0.0) -> [(text: String, logProb: Float)] {
        let gIds = word.compactMap { graphemeToId[$0] }
        guard !gIds.isEmpty, gIds.count + 2 <= Self.maxPos else { return [] }

        let encIn = [Self.bosId] + gIds + [Self.eosId]
        let encOut = encode(encIn)
        let d = Self.d
        let cK = linear(encOut, rows: encIn.count, wt: dCKW, b: dCKB, outD: d)
        let cV = linear(encOut, rows: encIn.count, wt: dCVW, b: dCVB, outD: d)
        let beams = beamSearchCore(cK: cK, cV: cV, encLen: encIn.count,
                                   beamWidth: beamWidth, lengthPenalty: lengthPenalty,
                                   diversityPenalty: diversityPenalty)

        return beams.compactMap { beam in
            let chars = beam.tokens.dropFirst()
                .filter { $0 >= 4 && $0 != Self.eosId }
                .compactMap { idToPhoneme[$0] }
            guard !chars.isEmpty else { return nil }
            return (text: String(chars), logProb: beam.logProb)
        }
    }

    // MARK: - Shared Helpers

    /// Token + positional embedding lookup.
    private func embed(_ ids: [Int], posW: [Float], lnW: [Float], lnB: [Float]) -> [Float] {
        let d = Self.d
        let n = ids.count
        var x = [Float](repeating: 0, count: n * d)
        sharedW.withUnsafeBufferPointer { sw in
            posW.withUnsafeBufferPointer { pw in
                x.withUnsafeMutableBufferPointer { xb in
                    for i in 0..<n {
                        vDSP_vadd(sw.baseAddress! + ids[i] * d, 1,
                                  pw.baseAddress! + (i + Self.posOff) * d, 1,
                                  xb.baseAddress! + i * d, 1,
                                  vDSP_Length(d))
                    }
                }
            }
        }
        layerNorm(&x, rows: n, w: lnW, b: lnB)
        return x
    }

    /// Embed a single token at a given position.
    private func embedOne(_ id: Int, pos: Int, posW: [Float], lnW: [Float], lnB: [Float]) -> [Float] {
        let d = Self.d
        var x = [Float](repeating: 0, count: d)
        sharedW.withUnsafeBufferPointer { sw in
            posW.withUnsafeBufferPointer { pw in
                x.withUnsafeMutableBufferPointer { xb in
                    vDSP_vadd(sw.baseAddress! + id * d, 1,
                              pw.baseAddress! + (pos + Self.posOff) * d, 1,
                              xb.baseAddress!, 1,
                              vDSP_Length(d))
                }
            }
        }
        layerNorm(&x, rows: 1, w: lnW, b: lnB)
        return x
    }

    // MARK: - Encoder

    private func encode(_ ids: [Int]) -> [Float] {
        let n = ids.count
        let x = embed(ids, posW: ePosW, lnW: eLnEW, lnB: eLnEB)

        var a = fullAttn(x, ctx: x, n: n, cLen: n, mask: nil,
                         qW: eQW, qB: eQB, kW: eKW, kB: eKB,
                         vW: eVW, vB: eVB, oW: eOW, oB: eOB)
        vDSP_vadd(a, 1, x, 1, &a, 1, vDSP_Length(a.count))
        layerNorm(&a, rows: n, w: eSaLnW, b: eSaLnB)

        var f = ffn(a, rows: n, w1: eF1W, b1: eF1B, w2: eF2W, b2: eF2B)
        vDSP_vadd(f, 1, a, 1, &f, 1, vDSP_Length(f.count))
        layerNorm(&f, rows: n, w: eFLnW, b: eFLnB)

        return f
    }

    // MARK: - Decoder (greedy)

    private func decode(cK: [Float], cV: [Float], encLen: Int) -> [Int] {
        var tokens = [Self.bosId]
        let maxSteps = min(50, Self.maxPos - 2)
        var saKCache = [Float]()
        var saVCache = [Float]()
        saKCache.reserveCapacity(maxSteps * Self.d)
        saVCache.reserveCapacity(maxSteps * Self.d)

        for step in 0..<maxSteps {
            let logits = decoderStep(token: tokens.last!, pos: step,
                                     saKCache: &saKCache, saVCache: &saVCache,
                                     cK: cK, cV: cV, encLen: encLen)

            var bestVal: Float = 0
            var bestIdx: vDSP_Length = 0
            vDSP_maxvi(logits, 1, &bestVal, &bestIdx, vDSP_Length(Self.vocabSize))
            let best = Int(bestIdx)
            if best == Self.eosId { break }
            tokens.append(best)
        }

        return tokens.dropFirst().filter { $0 >= 4 }
    }

    // MARK: - Decoder step helper

    /// Run one decoder step: given a token and position, update KV cache, return logits.
    private func decoderStep(
        token: Int, pos: Int,
        saKCache: inout [Float], saVCache: inout [Float],
        cK: [Float], cV: [Float], encLen: Int
    ) -> [Float] {
        let d = Self.d
        let x = embedOne(token, pos: pos, posW: dPosW, lnW: dLnEW, lnB: dLnEB)

        let newK = linear(x, rows: 1, wt: dSKW, b: dSKB, outD: d)
        let newV = linear(x, rows: 1, wt: dSVW, b: dSVB, outD: d)
        saKCache.append(contentsOf: newK)
        saVCache.append(contentsOf: newV)
        let n = saKCache.count / d

        let newQ = linear(x, rows: 1, wt: dSQW, b: dSQB, outD: d)
        var sa = sdpa(q: newQ, k: saKCache, v: saVCache, qLen: 1, kvLen: n, mask: nil)
        sa = linear(sa, rows: 1, wt: dSOW, b: dSOB, outD: d)
        vDSP_vadd(sa, 1, x, 1, &sa, 1, vDSP_Length(d))
        layerNorm(&sa, rows: 1, w: dSaLnW, b: dSaLnB)

        let cQ = linear(sa, rows: 1, wt: dCQW, b: dCQB, outD: d)
        var ca = sdpa(q: cQ, k: cK, v: cV, qLen: 1, kvLen: encLen, mask: nil)
        ca = linear(ca, rows: 1, wt: dCOW, b: dCOB, outD: d)
        vDSP_vadd(ca, 1, sa, 1, &ca, 1, vDSP_Length(d))
        layerNorm(&ca, rows: 1, w: dCaLnW, b: dCaLnB)

        var f = ffn(ca, rows: 1, w1: dF1W, b1: dF1B, w2: dF2W, b2: dF2B)
        vDSP_vadd(f, 1, ca, 1, &f, 1, vDSP_Length(d))
        layerNorm(&f, rows: 1, w: dFLnW, b: dFLnB)

        var logits = [Float](repeating: 0, count: Self.vocabSize)
        f.withUnsafeBufferPointer { fBuf in
            sharedW.withUnsafeBufferPointer { sw in
                logits.withUnsafeMutableBufferPointer { lb in
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                Int32(Self.vocabSize), Int32(d),
                                1.0, sw.baseAddress!, Int32(d),
                                fBuf.baseAddress!, 1,
                                0.0, lb.baseAddress!, 1)
                }
            }
        }
        vDSP_vadd(logits, 1, logitBias, 1, &logits, 1, vDSP_Length(Self.vocabSize))
        return logits
    }

    // MARK: - Decoder (beam search)

    private struct Beam {
        var tokens: [Int]
        var logProb: Float
        var saKCache: [Float]
        var saVCache: [Float]
        var finished: Bool
    }

    /// Core beam search returning all beams sorted best-first by length-normalized score.
    /// When `diversityPenalty > 0`, uses greedy diverse selection: at each step, candidates
    /// whose last token was already selected receive a penalty, encouraging token diversity.
    private func beamSearchCore(cK: [Float], cV: [Float], encLen: Int,
                                beamWidth: Int, lengthPenalty: Float,
                                diversityPenalty: Float = 0.0) -> [Beam] {
        let maxSteps = min(50, Self.maxPos - 2)
        let diverse = diversityPenalty > 0

        var beams = [Beam(tokens: [Self.bosId], logProb: 0.0,
                          saKCache: [], saVCache: [], finished: false)]

        for step in 0..<maxSteps {
            var candidates: [(tokens: [Int], logProb: Float,
                              saKCache: [Float], saVCache: [Float])] = []

            for beam in beams where !beam.finished {
                var kCache = beam.saKCache
                var vCache = beam.saVCache
                var logits = decoderStep(token: beam.tokens.last!, pos: step,
                                         saKCache: &kCache, saVCache: &vCache,
                                         cK: cK, cV: cV, encLen: encLen)
                logSoftmax(&logits)

                var indexed = logits.enumerated().map { ($0.offset, $0.element) }
                indexed.sort { $0.1 > $1.1 }
                for i in 0..<min(beamWidth, indexed.count) {
                    let (tokId, tokLogProb) = indexed[i]
                    guard tokLogProb.isFinite else { continue }
                    var newTokens = beam.tokens
                    newTokens.append(tokId)
                    candidates.append((tokens: newTokens,
                                       logProb: beam.logProb + tokLogProb,
                                       saKCache: kCache, saVCache: vCache))
                }
            }

            for beam in beams where beam.finished {
                candidates.append((tokens: beam.tokens, logProb: beam.logProb,
                                   saKCache: beam.saKCache, saVCache: beam.saVCache))
            }

            guard !candidates.isEmpty else { break }

            let chosen: [(tokens: [Int], logProb: Float,
                          saKCache: [Float], saVCache: [Float])]
            if diverse {
                // Greedy diverse selection: penalize repeated last-tokens
                let scored = candidates.map { c in
                    (c, c.logProb / powf(Float(c.tokens.count), lengthPenalty))
                }
                var selected: [(tokens: [Int], logProb: Float,
                               saKCache: [Float], saVCache: [Float])] = []
                var used = Set<Int>()
                var tokenCounts = [Int: Int]()

                while selected.count < beamWidth {
                    var bestIdx = -1
                    var bestScore: Float = -.infinity
                    for (idx, (c, baseScore)) in scored.enumerated() where !used.contains(idx) {
                        let penalty = Float(tokenCounts[c.tokens.last!] ?? 0) * diversityPenalty
                        let score = baseScore - penalty
                        if score > bestScore {
                            bestScore = score
                            bestIdx = idx
                        }
                    }
                    guard bestIdx >= 0 else { break }
                    used.insert(bestIdx)
                    let pick = scored[bestIdx].0
                    tokenCounts[pick.tokens.last!, default: 0] += 1
                    selected.append(pick)
                }
                chosen = selected
            } else {
                candidates.sort { ($0.logProb / powf(Float($0.tokens.count), lengthPenalty)) > ($1.logProb / powf(Float($1.tokens.count), lengthPenalty)) }
                chosen = Array(candidates.prefix(beamWidth))
            }

            beams = chosen.map { c in
                Beam(tokens: c.tokens, logProb: c.logProb,
                     saKCache: c.saKCache, saVCache: c.saVCache,
                     finished: c.tokens.last == Self.eosId)
            }

            if beams.allSatisfy({ $0.finished }) { break }
        }

        return beams.sorted {
            ($0.logProb / powf(Float($0.tokens.count), lengthPenalty)) >
            ($1.logProb / powf(Float($1.tokens.count), lengthPenalty))
        }
    }

    /// In-place log-softmax over vocabSize logits.
    private func logSoftmax(_ logits: inout [Float]) {
        let n = vDSP_Length(Self.vocabSize)
        var maxV: Float = 0
        vDSP_maxv(logits, 1, &maxV, n)
        var negMax = -maxV
        vDSP_vsadd(logits, 1, &negMax, &logits, 1, n)
        // Compute sum(exp(shifted)) without destroying shifted values
        var expBuf = logits
        var count = Int32(Self.vocabSize)
        vvexpf(&expBuf, expBuf, &count)
        var sum: Float = 0
        vDSP_sve(expBuf, 1, &sum, n)
        // logSoftmax = shifted - log(sum(exp(shifted)))
        var negLogSum = -logf(sum)
        vDSP_vsadd(logits, 1, &negLogSum, &logits, 1, n)
    }

    /// Phoneme-level Levenshtein edit distance.
    private static func phonemeEditDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        let m = a.count, n = b.count
        if m == 0 { return n }
        if n == 0 { return m }
        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)
        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                curr[j] = a[i-1] == b[j-1]
                    ? prev[j-1]
                    : 1 + min(prev[j-1], prev[j], curr[j-1])
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }

    // MARK: - Linear Algebra

    /// output = input @ W^T + bias.  W is [outD, inD] row-major.
    private func linear(_ input: [Float], rows: Int, wt: [Float], b: [Float],
                        outD: Int, inD: Int? = nil) -> [Float] {
        let k = inD ?? Self.d
        var out = [Float](repeating: 0, count: rows * outD)
        input.withUnsafeBufferPointer { inBuf in
            wt.withUnsafeBufferPointer { wBuf in
                out.withUnsafeMutableBufferPointer { oBuf in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(rows), Int32(outD), Int32(k),
                        1.0, inBuf.baseAddress!, Int32(k),
                        wBuf.baseAddress!, Int32(k),
                        0.0, oBuf.baseAddress!, Int32(outD))
                }
            }
        }
        b.withUnsafeBufferPointer { bBuf in
            out.withUnsafeMutableBufferPointer { oBuf in
                for r in 0..<rows {
                    vDSP_vadd(oBuf.baseAddress! + r * outD, 1,
                              bBuf.baseAddress!, 1,
                              oBuf.baseAddress! + r * outD, 1,
                              vDSP_Length(outD))
                }
            }
        }
        return out
    }

    /// Full attention: project Q/K/V, compute SDPA, project output.
    private func fullAttn(
        _ x: [Float], ctx: [Float], n: Int, cLen: Int, mask: [Float]?,
        qW: [Float], qB: [Float], kW: [Float], kB: [Float],
        vW: [Float], vB: [Float], oW: [Float], oB: [Float]
    ) -> [Float] {
        let d = Self.d
        let q = linear(x, rows: n, wt: qW, b: qB, outD: d)
        let k = linear(ctx, rows: cLen, wt: kW, b: kB, outD: d)
        let v = linear(ctx, rows: cLen, wt: vW, b: vB, outD: d)
        let a = sdpa(q: q, k: k, v: v, qLen: n, kvLen: cLen, mask: mask)
        return linear(a, rows: n, wt: oW, b: oB, outD: d)
    }

    /// Scaled dot-product attention.
    private func sdpa(q: [Float], k: [Float], v: [Float],
                      qLen: Int, kvLen: Int, mask: [Float]?) -> [Float] {
        let d = Self.d
        var scores = [Float](repeating: 0, count: qLen * kvLen)
        q.withUnsafeBufferPointer { qBuf in
            k.withUnsafeBufferPointer { kBuf in
                scores.withUnsafeMutableBufferPointer { sBuf in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(qLen), Int32(kvLen), Int32(d),
                        Self.scale, qBuf.baseAddress!, Int32(d),
                        kBuf.baseAddress!, Int32(d),
                        0.0, sBuf.baseAddress!, Int32(kvLen))
                }
            }
        }

        if let mask {
            vDSP_vadd(scores, 1, mask, 1, &scores, 1, vDSP_Length(scores.count))
        }

        // Softmax per row
        scores.withUnsafeMutableBufferPointer { sBuf in
            for r in 0..<qLen {
                let row = sBuf.baseAddress! + r * kvLen
                let len = vDSP_Length(kvLen)
                var maxV: Float = 0
                vDSP_maxv(row, 1, &maxV, len)
                var negMax = -maxV
                vDSP_vsadd(row, 1, &negMax, row, 1, len)
                // exp each element
                var count = Int32(kvLen)
                vvexpf(row, row, &count)
                var sum: Float = 0
                vDSP_sve(row, 1, &sum, len)
                if sum > 0 { vDSP_vsdiv(row, 1, &sum, row, 1, len) }
            }
        }

        // output = weights @ V
        var out = [Float](repeating: 0, count: qLen * d)
        scores.withUnsafeBufferPointer { sBuf in
            v.withUnsafeBufferPointer { vBuf in
                out.withUnsafeMutableBufferPointer { oBuf in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(qLen), Int32(d), Int32(kvLen),
                        1.0, sBuf.baseAddress!, Int32(kvLen),
                        vBuf.baseAddress!, Int32(d),
                        0.0, oBuf.baseAddress!, Int32(d))
                }
            }
        }
        return out
    }

    /// FFN: gelu(x @ W1^T + b1) @ W2^T + b2
    private func ffn(_ input: [Float], rows: Int,
                     w1: [Float], b1: [Float], w2: [Float], b2: [Float]) -> [Float] {
        var h = linear(input, rows: rows, wt: w1, b: b1, outD: Self.ffnDim)
        let invSqrt2: Float = 1.0 / sqrtf(2.0)
        for i in 0..<h.count { h[i] = h[i] * 0.5 * (1.0 + erff(h[i] * invSqrt2)) }
        return linear(h, rows: rows, wt: w2, b: b2, outD: Self.d, inD: Self.ffnDim)
    }

    /// Layer normalization over last dimension (d_model).
    private func layerNorm(_ x: inout [Float], rows: Int, w: [Float], b: [Float]) {
        let d = Self.d
        let eps: Float = 1e-5
        x.withUnsafeMutableBufferPointer { xBuf in
            w.withUnsafeBufferPointer { wBuf in
                b.withUnsafeBufferPointer { bBuf in
                    for r in 0..<rows {
                        let row = xBuf.baseAddress! + r * d
                        let len = vDSP_Length(d)
                        var mean: Float = 0
                        vDSP_meanv(row, 1, &mean, len)
                        var negMean = -mean
                        vDSP_vsadd(row, 1, &negMean, row, 1, len)
                        var variance: Float = 0
                        vDSP_svesq(row, 1, &variance, len)
                        variance /= Float(d)
                        var inv = 1.0 / sqrtf(variance + eps)
                        vDSP_vsmul(row, 1, &inv, row, 1, len)
                        // x = x * w + b
                        vDSP_vma(row, 1, wBuf.baseAddress!, 1,
                                 bBuf.baseAddress!, 1, row, 1, len)
                    }
                }
            }
        }
    }
}

/// Phoneme trigram language model for rescoring beam candidates.
/// Trained on CMUdict IPA transcriptions (~117K words).
struct PhonemeTrigramLM: Sendable {
    // Flattened trigram: key = 2-char context string, value = dict of next-char counts
    private let counts: [String: [Character: Int]]
    private let ctxTotals: [String: Int]
    private let vocabSize: Int

    init?(url: URL) {
        guard let data = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var counts = [String: [Character: Int]]()
        var ctxTotals = [String: Int]()
        var V = 42

        for line in data.split(separator: "\n") {
            if line.hasPrefix("V=") {
                V = Int(line.dropFirst(2)) ?? 42
                continue
            }
            let parts = line.split(separator: "\t")
            guard parts.count == 3 else { continue }
            let ctx = String(parts[0])
            guard let nxt = parts[1].first, let count = Int(parts[2]) else { continue }
            counts[ctx, default: [:]][nxt] = count
            ctxTotals[ctx, default: 0] += count
        }

        self.counts = counts
        self.ctxTotals = ctxTotals
        self.vocabSize = V
    }

    /// Log-probability of a phoneme sequence under the trigram model.
    func logProb(_ ipa: String) -> Float {
        let normalized: [Character] = ipa.compactMap { c -> Character? in
            switch c {
            case "ˈ", "ˌ", "ʔ": return nil
            case "ɾ": return "t"
            case "ᵊ": return "ə"
            case "ᵻ": return "ɪ"
            default: return c
            }
        }
        let chars: [Character] = ["^", "^"] + normalized + ["$"]
        var lp: Float = 0
        for i in 0..<chars.count - 2 {
            let ctx = String([chars[i], chars[i + 1]])
            let count = counts[ctx]?[chars[i + 2]] ?? 0
            let ctxTotal = ctxTotals[ctx] ?? 0
            lp += logf(Float(count + 1) / Float(ctxTotal + vocabSize))
        }
        return lp
    }
}
