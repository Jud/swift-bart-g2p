import Foundation
import BARTG2P

/// Normalize IPA for loose comparison: strip stress, normalize allophones.
func normalize(_ ipa: String) -> String {
    var s = ipa
    s = s.replacingOccurrences(of: "ˈ", with: "")
    s = s.replacingOccurrences(of: "ˌ", with: "")
    s = s.replacingOccurrences(of: "ɾ", with: "t")
    s = s.replacingOccurrences(of: "ᵊ", with: "ə")
    s = s.replacingOccurrences(of: "ᵻ", with: "ɪ")
    s = s.replacingOccurrences(of: "ʔ", with: "")
    return s
}

guard CommandLine.arguments.count >= 3 else {
    print("Usage: GenerateRerankerData <cmudict_full_ipa.tsv> <output.tsv>")
    exit(1)
}

let dictPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]

guard let bart = BARTG2P.fromBundle() else {
    print("ERROR: Failed to load BART model from bundle")
    exit(1)
}

guard let dictContent = try? String(contentsOfFile: dictPath, encoding: .utf8) else {
    print("ERROR: Could not read \(dictPath)")
    exit(1)
}

let lines = dictContent.split(separator: "\n").filter { !$0.isEmpty }
let total = lines.count
print("Loaded \(total) words from \(dictPath)")

guard let fileHandle = FileHandle(forWritingAtPath: outputPath) ?? {
    FileManager.default.createFile(atPath: outputPath, contents: nil)
    return FileHandle(forWritingAtPath: outputPath)
}() else {
    print("ERROR: Could not open \(outputPath) for writing")
    exit(1)
}

var processed = 0
let startTime = CFAbsoluteTimeGetCurrent()

for line in lines {
    let parts = line.split(separator: "\t", maxSplits: 1)
    guard parts.count == 2 else { continue }
    let word = String(parts[0])
    let reference = String(parts[1])

    let beamWidth = CommandLine.arguments.count >= 4 ? Int(CommandLine.arguments[3]) ?? 8 : 8
    let candidates = bart.predictNBestScored(word, beamWidth: beamWidth)
    for (text, modelLP) in candidates {
        let trigramLP = bart.trigramScore(text)
        let isCorrect = normalize(text) == normalize(reference) ? 1 : 0
        let row = "\(word)\t\(text)\t\(modelLP)\t\(trigramLP)\t\(isCorrect)\n"
        fileHandle.write(row.data(using: .utf8)!)
    }

    processed += 1
    if processed % 5000 == 0 {
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let rate = Double(processed) / elapsed
        let eta = Double(total - processed) / rate
        print("  \(processed)/\(total) (\(String(format: "%.1f", Double(processed)/Double(total)*100))%) "
            + "\(String(format: "%.0f", rate)) words/s, ETA \(String(format: "%.0f", eta))s")
    }
}

fileHandle.closeFile()
let elapsed = CFAbsoluteTimeGetCurrent() - startTime
print("Done: \(processed) words in \(String(format: "%.1f", elapsed))s → \(outputPath)")
