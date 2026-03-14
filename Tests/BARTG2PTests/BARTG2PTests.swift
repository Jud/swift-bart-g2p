import Foundation
import Testing
@testable import BARTG2P

@Suite("BARTG2P")
struct BARTG2PTests {
    @Test("BART loads from bundle")
    func loads() {
        #expect(BARTG2P.fromBundle() != nil)
    }

    @Test("50-word greedy reference set matches expected IPA")
    func referenceMatch() throws {
        let bart = try #require(BARTG2P.fromBundle())
        let expected: [(String, String)] = [
            ("hello", "hˈɛlO"),
            ("cat", "kˈæt"),
            ("world", "wˈɜɹld"),
            ("kubernetes", "kˌubəɹnˈits"),
            ("apple", "əpˈAl"),
            ("banana", "bənˈɑnə"),
            ("computer", "kəmpjˈuɾəɹ"),
            ("telephone", "tˈɛləfˌOn"),
            ("beautiful", "bjˈuɾəfᵊl"),
            ("strange", "stɹˈAnʤ"),
            ("through", "θɹˈu"),
            ("thought", "θˈɔt"),
            ("enough", "ɪnˈʌf"),
            ("cough", "kˈɔf"),
            ("dough", "dˈO"),
            ("tough", "tˈʌf"),
            ("knight", "nˈIt"),
            ("psychology", "sIkˈɑləʤi"),
            ("pneumonia", "numˈOniə"),
            ("gnome", "nˈOm"),
            ("algorithm", "ˈælɡəɹˌɪðəm"),
            ("symphony", "sˈɪmfəni"),
            ("architecture", "ˌɑɹkətˈɛkʧəɹ"),
            ("Mississippi", "mˌɪsəsˈɪpi"),
            ("xylophone", "zˈIləfˌOn"),
            ("rhythm", "ɹˈɪðəm"),
            ("queue", "kjˈu"),
            ("colonel", "kˈɑlənᵊl"),
            ("Wednesday", "wˈɛdnzdˌA"),
            ("February", "fˈɛbjəwˌɛɹi"),
            ("library", "lˈIbɹɛɹi"),
            ("comfortable", "kˈʌmfəɹɾəbᵊl"),
            ("temperature", "tˈɛmpəɹəʧˌʊɹ"),
            ("chocolate", "ʧˈɑkəlˌAt"),
            ("vegetable", "vˈɛʤtəbᵊl"),
            ("interesting", "ˌɪntɹəstˈɪŋ"),
            ("different", "dˈɪfəɹənt"),
            ("restaurant", "ɹəstˈɔɹənt"),
            ("mountain", "mˈWntᵊn"),
            ("language", "lˈæŋɡwɪʤ"),
            ("important", "ɪmpˈɔɹtᵊnt"),
            ("question", "kwˈɛsʧən"),
            ("practice", "pɹˈæktəs"),
            ("natural", "nˈæʧəɹᵊl"),
            ("country", "kˈWntɹi"),
            ("sentence", "sˈɛntᵊns"),
            ("remember", "ɹᵻmˈɛmbəɹ"),
            ("together", "təɡˈɛðəɹ"),
            ("example", "ɪɡzˈæmpᵊl"),
            ("between", "bᵻtwˈin"),
            ("problem", "pɹˈɑbləm"),
        ]
        var failures = 0
        for (word, ref) in expected {
            let result = bart.predict(word)
            if result != ref {
                failures += 1
                Issue.record("BART(\(word)): got \(result ?? "nil"), expected \(ref)")
            }
        }
        #expect(failures == 0, "\(failures)/\(expected.count) words mismatched")
    }

    @Test("Edge cases: empty and too-long inputs")
    func edgeCases() throws {
        let bart = try #require(BARTG2P.fromBundle())
        #expect(bart.predict("") == nil)
        let tooLong = String(repeating: "a", count: 100)
        #expect(bart.predict(tooLong) == nil)
    }

    @Test("All output characters are valid IPA phonemes")
    func validPhonemes() throws {
        let bart = try #require(BARTG2P.fromBundle())
        let words = ["hello", "strange", "psychology", "xylophone", "comfortable"]
        for word in words {
            if let result = bart.predict(word) {
                for ch in result {
                    #expect(!ch.isASCII || ch.isLetter || ch == "ˈ" || ch == "ˌ",
                            "Unexpected character '\(ch)' (U+\(String(ch.unicodeScalars.first!.value, radix: 16))) in output for '\(word)'")
                }
            }
        }
    }

    @Test("Dictionary accuracy: 1000-word CMUdict sample (greedy)")
    func dictAccuracyGreedy() throws {
        let bart = try #require(BARTG2P.fromBundle())
        let r = measureDictAccuracy(bart: bart) { bart, word in bart.predict(word) }
        let exactPct = Double(r.exact) / Double(r.total) * 100.0
        let loosePct = Double(r.loose) / Double(r.total) * 100.0
        print("dict_accuracy_greedy: exact=\(r.exact)/\(r.total) (\(String(format: "%.1f", exactPct))%) loose=\(r.loose)/\(r.total) (\(String(format: "%.1f", loosePct))%) PER=\(String(format: "%.1f", r.avgPER * 100))%")
        #expect(r.loose >= 490, "Greedy loose accuracy regressed below 49%")
        #expect(r.avgPER < 0.14, "Greedy PER regressed above 14%")
    }

    @Test("Dictionary accuracy: normalization level analysis")
    func dictAccuracyNormLevels() throws {
        let bart = try #require(BARTG2P.fromBundle())
        guard let url = Bundle.module.url(forResource: "cmudict_ref_1000", withExtension: "tsv"),
              let data = try? String(contentsOf: url, encoding: .utf8)
        else { return }

        var total = 0, exact = 0, loose = 0, dialect = 0
        var loosePER: Float = 0, dialectPER: Float = 0

        for line in data.split(separator: "\n") where !line.isEmpty {
            let parts = line.split(separator: "\t", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let word = String(parts[0])
            let expected = String(parts[1])
            total += 1
            guard let result = bart.predict(word) else {
                loosePER += 1.0; dialectPER += 1.0; continue
            }
            if result == expected { exact += 1; loose += 1; dialect += 1 }
            else if normalize(result) == normalize(expected) { loose += 1; dialect += 1 }
            else if dialectNormalize(result) == dialectNormalize(expected) { dialect += 1 }
            loosePER += phonemeErrorRate(normalize(result), normalize(expected))
            dialectPER += phonemeErrorRate(dialectNormalize(result), dialectNormalize(expected))
        }

        print("normalization_levels (greedy, n=\(total)):")
        print("  exact:   \(exact)/\(total) (\(String(format: "%.1f", Double(exact)/Double(total)*100))%)")
        print("  loose:   \(loose)/\(total) (\(String(format: "%.1f", Double(loose)/Double(total)*100))%) PER=\(String(format: "%.1f", loosePER/Float(total)*100))%")
        print("  dialect: \(dialect)/\(total) (\(String(format: "%.1f", Double(dialect)/Double(total)*100))%) PER=\(String(format: "%.1f", dialectPER/Float(total)*100))%")
    }

    @Test("Oracle upper bound: best of N beams vs CMUdict")
    func oracleUpperBound() throws {
        let bart = try #require(BARTG2P.fromBundle())
        guard let url = Bundle.module.url(forResource: "cmudict_ref_1000", withExtension: "tsv"),
              let data = try? String(contentsOf: url, encoding: .utf8)
        else { return }

        var total = 0, greedyLoose = 0, oracleLoose = 0, oracleExact = 0

        for line in data.split(separator: "\n") where !line.isEmpty {
            let parts = line.split(separator: "\t", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let word = String(parts[0])
            let expected = String(parts[1])
            total += 1

            // Greedy
            if let g = bart.predict(word), normalize(g) == normalize(expected) {
                greedyLoose += 1
            }

            // Oracle: check if ANY beam matches
            let candidates = bart.predictNBest(word, beamWidth: 8)
            var foundExact = false, foundLoose = false
            for c in candidates {
                if c == expected { foundExact = true; foundLoose = true; break }
                if normalize(c) == normalize(expected) { foundLoose = true }
            }
            if foundExact { oracleExact += 1 }
            if foundLoose { oracleLoose += 1 }
        }

        print("oracle_beam8: greedy_loose=\(greedyLoose)/\(total) (\(String(format: "%.1f", Double(greedyLoose)/Double(total)*100))%) oracle_exact=\(oracleExact)/\(total) (\(String(format: "%.1f", Double(oracleExact)/Double(total)*100))%) oracle_loose=\(oracleLoose)/\(total) (\(String(format: "%.1f", Double(oracleLoose)/Double(total)*100))%)")
    }

    @Test("Dictionary accuracy: rescoreLM mode")
    func dictAccuracyRescoreLM() throws {
        let bart = try #require(BARTG2P.fromBundle())
        let r = measureDictAccuracy(bart: bart) { bart, word in bart.predict(word, rescoreLM: true) }
        let loosePct = Double(r.loose) / Double(r.total) * 100.0
        print("dict_accuracy_rescoreLM: exact=\(r.exact)/\(r.total) (\(String(format: "%.1f", Double(r.exact)/Double(r.total)*100))%) loose=\(r.loose)/\(r.total) (\(String(format: "%.1f", loosePct))%) PER=\(String(format: "%.1f", r.avgPER * 100))%")
        #expect(r.loose >= 510, "RescoreLM loose accuracy regressed below 51%")
        #expect(r.avgPER < 0.13, "RescoreLM PER regressed above 13%")
    }

    @Test("Benchmark: inference throughput")
    func benchmark() throws {
        let bart = try #require(BARTG2P.fromBundle())
        let words = [
            "hello", "cat", "world", "kubernetes", "apple", "banana",
            "computer", "telephone", "beautiful", "strange", "through",
            "thought", "enough", "cough", "dough", "tough", "knight",
            "psychology", "pneumonia", "gnome", "algorithm", "symphony",
            "architecture", "Mississippi", "xylophone", "rhythm",
            "queue", "colonel", "Wednesday", "February", "library",
            "comfortable", "temperature", "chocolate", "vegetable",
            "interesting", "different", "restaurant", "mountain",
            "language", "important", "question", "practice",
            "natural", "country", "sentence", "remember",
            "together", "example", "between", "problem",
        ]

        // Warmup
        for word in words { _ = bart.predict(word) }

        // Timed run (3 iterations for stability)
        let iterations = 3
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            for word in words {
                _ = bart.predict(word)
            }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let totalPredictions = words.count * iterations
        let avgMs = (elapsed / Double(totalPredictions)) * 1000.0
        let throughput = Double(totalPredictions) / elapsed

        print("perf: avg_ms=\(String(format: "%.3f", avgMs)) throughput=\(String(format: "%.1f", throughput))words/s total=\(String(format: "%.3f", elapsed))s (\(totalPredictions) predictions)")
    }
}

/// Accuracy results from dictionary comparison.
struct DictAccuracyResult {
    var exact: Int
    var loose: Int
    var total: Int
    var avgPER: Float
    var mismatches: [(String, String, String)]
}

/// Measure dictionary accuracy using a prediction closure.
private func measureDictAccuracy(bart: BARTG2P, predict: (BARTG2P, String) -> String?) -> DictAccuracyResult {
    guard let url = Bundle.module.url(forResource: "cmudict_ref_1000", withExtension: "tsv"),
          let data = try? String(contentsOf: url, encoding: .utf8)
    else { return DictAccuracyResult(exact: 0, loose: 0, total: 0, avgPER: 1.0, mismatches: []) }

    var total = 0, exactMatch = 0, looseMatch = 0
    var mismatches: [(String, String, String)] = []
    var totalPER: Float = 0

    for line in data.split(separator: "\n") where !line.isEmpty {
        let parts = line.split(separator: "\t", maxSplits: 1)
        guard parts.count == 2 else { continue }
        let word = String(parts[0])
        let expected = String(parts[1])
        total += 1

        guard let result = predict(bart, word) else {
            totalPER += 1.0
            continue
        }

        let normResult = normalize(result)
        let normExpected = normalize(expected)
        totalPER += phonemeErrorRate(normResult, normExpected)

        if result == expected {
            exactMatch += 1
            looseMatch += 1
        } else if normResult == normExpected {
            looseMatch += 1
        } else if mismatches.count < 20 {
            mismatches.append((word, expected, result))
        }
    }
    let avgPER = total > 0 ? totalPER / Float(total) : 1.0
    return DictAccuracyResult(exact: exactMatch, loose: looseMatch, total: total,
                              avgPER: avgPER, mismatches: mismatches)
}

/// Normalize IPA for loose comparison: strip stress, normalize allophones.
private func normalize(_ ipa: String) -> String {
    var s = ipa
    // Strip stress markers
    s = s.replacingOccurrences(of: "ˈ", with: "")
    s = s.replacingOccurrences(of: "ˌ", with: "")
    // Normalize model-specific allophones to CMUdict equivalents
    s = s.replacingOccurrences(of: "ɾ", with: "t")  // flap → /t/
    s = s.replacingOccurrences(of: "ᵊ", with: "ə")  // reduced schwa → schwa
    s = s.replacingOccurrences(of: "ᵻ", with: "ɪ")  // barred-i → /ɪ/
    s = s.replacingOccurrences(of: "ʔ", with: "")    // glottal stop (not in CMUdict)
    return s
}

/// Dialect-aware normalization: merge vowels that are commonly conflated.
private func dialectNormalize(_ ipa: String) -> String {
    var s = normalize(ipa)
    // Cot-caught merger: ɑ↔ɔ (most American English dialects)
    s = s.replacingOccurrences(of: "ɔ", with: "ɑ")
    // Unstressed vowel reduction: æ→ə (very common in fast speech)
    s = s.replacingOccurrences(of: "æ", with: "ə")
    // Unstressed ʌ→ə (these are the same phoneme in unstressed positions)
    s = s.replacingOccurrences(of: "ʌ", with: "ə")
    return s
}

/// Phoneme error rate: Levenshtein edit distance / reference length.
private func phonemeErrorRate(_ hypothesis: String, _ reference: String) -> Float {
    let h = Array(hypothesis)
    let r = Array(reference)
    let m = h.count, n = r.count
    if n == 0 { return m == 0 ? 0 : 1.0 }
    if m == 0 { return 1.0 }
    var dp = Array(0...n)
    for i in 1...m {
        var prev = dp[0]
        dp[0] = i
        for j in 1...n {
            let tmp = dp[j]
            if h[i-1] == r[j-1] {
                dp[j] = prev
            } else {
                dp[j] = min(prev, dp[j], dp[j-1]) + 1
            }
            prev = tmp
        }
    }
    return Float(dp[n]) / Float(n)
}
