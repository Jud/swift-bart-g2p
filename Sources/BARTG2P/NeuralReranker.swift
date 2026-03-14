import Accelerate
import Foundation

/// 1D CNN neural reranker for scoring (grapheme, phoneme) candidate pairs.
///
/// Architecture (~36K params):
///   char_embed(83x32) + type_embed(3x32) → conv1(k=5, 32→64, BN, ReLU)
///   → conv2(k=5, 64→64, BN, ReLU) → global_avg_pool → concat(model_lp, trigram_lp)
///   → fc1(66→32, ReLU) → fc2(32→1)
struct NeuralReranker: Sendable {
    private static let vocabSize = 83
    private static let embedDim = 32
    private static let conv1Out = 64
    private static let conv2Out = 64
    private static let kernelSize = 5
    private static let fcHidden = 32
    private static let padId = 0
    private static let bosId = 1
    private static let sepId = 2
    private static let eosId = 3

    private let charToId: [Character: Int]

    // Weights
    private let charEmbedW: [Float]   // [83, 32]
    private let typeEmbedW: [Float]   // [3, 32]
    private let conv1W: [Float]       // [64, 5*32] (transposed from PyTorch [64,32,5])
    private let conv1B: [Float]       // [64]
    private let bn1Scale: [Float]     // [64] fused: gamma / sqrt(var + eps)
    private let bn1Bias: [Float]      // [64] fused: beta - mean * scale
    private let conv2W: [Float]       // [64, 5*64]
    private let conv2B: [Float]       // [64]
    private let bn2Scale: [Float]     // [64]
    private let bn2Bias: [Float]      // [64]
    private let fc1W: [Float]         // [32, 66]
    private let fc1B: [Float]         // [32]
    private let fc2W: [Float]         // [1, 32]
    private let fc2B: [Float]         // [1]
    private let featMean: [Float]     // [2] model_lp, trigram_lp means
    private let featStd: [Float]      // [2] model_lp, trigram_lp stds

    init?(url: URL) {
        let chars: [Character] = [
            "'", "-", ".",
            "A","B","C","D","E","F","G","H","I","J","K","L","M",
            "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
            "a","b","c","d","e","f","g","h","i","j","k","l","m",
            "n","o","p","q","r","s","t","u","v","w","x","y","z",
            "æ","ð","ŋ","θ","ɑ","ɔ","ə","ɛ","ɜ","ɡ","ɪ","ɹ","ɾ",
            "ʃ","ʊ","ʌ","ʒ","ʔ","ʤ","ʧ","ˈ","ˌ","ᵊ","ᵻ",
        ]
        var mapping = [Character: Int]()
        for (i, c) in chars.enumerated() { mapping[c] = i + 4 }
        self.charToId = mapping

        guard let st = try? SafetensorsLoader(url: url) else { return nil }
        do {
            charEmbedW = try st.floats(for: "char_embed.weight")
            typeEmbedW = try st.floats(for: "type_embed.weight")

            // Transpose conv weights from [outCh, inCh, K] to [outCh, K*inCh]
            conv1W = Self.transposeConvWeight(
                try st.floats(for: "conv1.weight"), outCh: 64, inCh: 32, kernel: 5)
            conv1B = try st.floats(for: "conv1.bias")
            conv2W = Self.transposeConvWeight(
                try st.floats(for: "conv2.weight"), outCh: 64, inCh: 64, kernel: 5)
            conv2B = try st.floats(for: "conv2.bias")

            // Fuse batch norm: scale = gamma/sqrt(var+eps), bias = beta - mean*scale
            let eps: Float = 1e-5
            let bn1G = try st.floats(for: "bn1.weight")
            let bn1Beta = try st.floats(for: "bn1.bias")
            let bn1M = try st.floats(for: "bn1.running_mean")
            let bn1V = try st.floats(for: "bn1.running_var")
            var s1 = [Float](repeating: 0, count: 64)
            var b1 = [Float](repeating: 0, count: 64)
            for i in 0..<64 {
                s1[i] = bn1G[i] / sqrtf(bn1V[i] + eps)
                b1[i] = bn1Beta[i] - bn1M[i] * s1[i]
            }
            bn1Scale = s1; bn1Bias = b1

            let bn2G = try st.floats(for: "bn2.weight")
            let bn2Beta = try st.floats(for: "bn2.bias")
            let bn2M = try st.floats(for: "bn2.running_mean")
            let bn2V = try st.floats(for: "bn2.running_var")
            var s2 = [Float](repeating: 0, count: 64)
            var b2 = [Float](repeating: 0, count: 64)
            for i in 0..<64 {
                s2[i] = bn2G[i] / sqrtf(bn2V[i] + eps)
                b2[i] = bn2Beta[i] - bn2M[i] * s2[i]
            }
            bn2Scale = s2; bn2Bias = b2

            fc1W = try st.floats(for: "fc1.weight")
            fc1B = try st.floats(for: "fc1.bias")
            fc2W = try st.floats(for: "fc2.weight")
            fc2B = try st.floats(for: "fc2.bias")
            featMean = (try? st.floats(for: "feat_mean")) ?? [0, 0]
            featStd = (try? st.floats(for: "feat_std")) ?? [1, 1]
        } catch {
            return nil
        }
    }

    /// Score a (word, candidate) pair. Higher = more likely correct.
    func score(word: String, candidate: String,
               modelLogProb: Float, trigramLogProb: Float) -> Float {
        // Build input: [BOS] graphemes [SEP] phonemes [EOS]
        var tokenIds = [Self.bosId]
        var typeIds: [Int] = [2]  // special

        for c in word.lowercased() {
            tokenIds.append(charToId[c] ?? Self.padId)
            typeIds.append(0)  // grapheme
        }
        tokenIds.append(Self.sepId)
        typeIds.append(2)  // special

        for c in candidate {
            tokenIds.append(charToId[c] ?? Self.padId)
            typeIds.append(1)  // phoneme
        }
        tokenIds.append(Self.eosId)
        typeIds.append(2)  // special

        let seqLen = tokenIds.count
        let d = Self.embedDim

        // Embed: char_embed[token] + type_embed[type]
        var x = [Float](repeating: 0, count: seqLen * d)
        charEmbedW.withUnsafeBufferPointer { ce in
            typeEmbedW.withUnsafeBufferPointer { te in
                x.withUnsafeMutableBufferPointer { xb in
                    for i in 0..<seqLen {
                        vDSP_vadd(ce.baseAddress! + tokenIds[i] * d, 1,
                                  te.baseAddress! + typeIds[i] * d, 1,
                                  xb.baseAddress! + i * d, 1,
                                  vDSP_Length(d))
                    }
                }
            }
        }

        // Conv1 + fused BN + ReLU: [seqLen, 32] → [seqLen, 64]
        var h = conv1d(x, seqLen: seqLen, inCh: d, outCh: Self.conv1Out,
                       kernel: Self.kernelSize, w: conv1W, b: conv1B)
        fusedBNReLU(&h, len: seqLen * Self.conv1Out, ch: Self.conv1Out,
                    scale: bn1Scale, bias: bn1Bias)

        // Conv2 + fused BN + ReLU: [seqLen, 64] → [seqLen, 64]
        h = conv1d(h, seqLen: seqLen, inCh: Self.conv1Out, outCh: Self.conv2Out,
                   kernel: Self.kernelSize, w: conv2W, b: conv2B)
        fusedBNReLU(&h, len: seqLen * Self.conv2Out, ch: Self.conv2Out,
                    scale: bn2Scale, bias: bn2Bias)

        // Global average pool: [seqLen, 64] → [64]
        var pooled = [Float](repeating: 0, count: Self.conv2Out)
        h.withUnsafeBufferPointer { hBuf in
            pooled.withUnsafeMutableBufferPointer { pBuf in
                for c in 0..<Self.conv2Out {
                    vDSP_meanv(hBuf.baseAddress! + c, vDSP_Stride(Self.conv2Out),
                               &pBuf[c], vDSP_Length(seqLen))
                }
            }
        }

        // Concat: [64] + [normalized modelLP, trigramLP] → [66]
        var features = pooled
        features.append((modelLogProb - featMean[0]) / featStd[0])
        features.append((trigramLogProb - featMean[1]) / featStd[1])

        // FC1: 66 → 32 + ReLU
        var fc1Out = linearFC(features, w: fc1W, b: fc1B, outD: Self.fcHidden, inD: 66)
        var zero: Float = 0
        vDSP_vthres(fc1Out, 1, &zero, &fc1Out, 1, vDSP_Length(Self.fcHidden))

        // FC2: 32 → 1
        let fc2Out = linearFC(fc1Out, w: fc2W, b: fc2B, outD: 1, inD: Self.fcHidden)
        return fc2Out[0]
    }

    // MARK: - Ops

    /// Transpose conv weight from PyTorch [outCh, inCh, K] to [outCh, K*inCh] for im2col.
    private static func transposeConvWeight(
        _ w: [Float], outCh: Int, inCh: Int, kernel: Int
    ) -> [Float] {
        let colDim = kernel * inCh
        var out = [Float](repeating: 0, count: outCh * colDim)
        for o in 0..<outCh {
            for c in 0..<inCh {
                for k in 0..<kernel {
                    out[o * colDim + k * inCh + c] = w[o * (inCh * kernel) + c * kernel + k]
                }
            }
        }
        return out
    }

    /// Conv1D with same-padding via im2col + SGEMM.
    /// Input: [seqLen, inCh], Weight: [outCh, K*inCh], Output: [seqLen, outCh].
    private func conv1d(_ input: [Float], seqLen: Int, inCh: Int, outCh: Int,
                        kernel: Int, w: [Float], b: [Float]) -> [Float] {
        let pad = kernel / 2
        let colDim = inCh * kernel
        var cols = [Float](repeating: 0, count: seqLen * colDim)

        input.withUnsafeBufferPointer { inBuf in
            cols.withUnsafeMutableBufferPointer { colBuf in
                for t in 0..<seqLen {
                    for k in 0..<kernel {
                        let inPos = t - pad + k
                        guard inPos >= 0 && inPos < seqLen else { continue }
                        let srcOff = inPos * inCh
                        let dstOff = t * colDim + k * inCh
                        memcpy(colBuf.baseAddress! + dstOff,
                               inBuf.baseAddress! + srcOff,
                               inCh * MemoryLayout<Float>.size)
                    }
                }
            }
        }

        var out = [Float](repeating: 0, count: seqLen * outCh)
        cols.withUnsafeBufferPointer { colBuf in
            w.withUnsafeBufferPointer { wBuf in
                out.withUnsafeMutableBufferPointer { oBuf in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                Int32(seqLen), Int32(outCh), Int32(colDim),
                                1.0, colBuf.baseAddress!, Int32(colDim),
                                wBuf.baseAddress!, Int32(colDim),
                                0.0, oBuf.baseAddress!, Int32(outCh))
                }
            }
        }

        // Add bias per row
        b.withUnsafeBufferPointer { bBuf in
            out.withUnsafeMutableBufferPointer { oBuf in
                for t in 0..<seqLen {
                    vDSP_vadd(oBuf.baseAddress! + t * outCh, 1,
                              bBuf.baseAddress!, 1,
                              oBuf.baseAddress! + t * outCh, 1,
                              vDSP_Length(outCh))
                }
            }
        }
        return out
    }

    /// Fused BatchNorm (inference) + ReLU, applied per-channel in [seqLen, ch] layout.
    private func fusedBNReLU(_ x: inout [Float], len: Int, ch: Int,
                             scale: [Float], bias: [Float]) {
        let seqLen = len / ch
        x.withUnsafeMutableBufferPointer { xBuf in
            for c in 0..<ch {
                let s = scale[c]
                let b = bias[c]
                for t in 0..<seqLen {
                    let idx = t * ch + c
                    let v = xBuf[idx] * s + b
                    xBuf[idx] = v > 0 ? v : 0
                }
            }
        }
    }

    /// output = input @ W^T + bias. W is [outD, inD].
    private func linearFC(_ input: [Float], w: [Float], b: [Float],
                          outD: Int, inD: Int) -> [Float] {
        var out = [Float](repeating: 0, count: outD)
        input.withUnsafeBufferPointer { inBuf in
            w.withUnsafeBufferPointer { wBuf in
                out.withUnsafeMutableBufferPointer { oBuf in
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                Int32(outD), Int32(inD),
                                1.0, wBuf.baseAddress!, Int32(inD),
                                inBuf.baseAddress!, 1,
                                0.0, oBuf.baseAddress!, 1)
                }
            }
        }
        vDSP_vadd(out, 1, b, 1, &out, 1, vDSP_Length(outD))
        return out
    }
}
