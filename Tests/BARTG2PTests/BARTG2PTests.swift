import Foundation
import Testing
@testable import BARTG2P

@Suite("BARTG2P")
struct BARTG2PTests {
    @Test("BART loads from bundle")
    func loads() {
        #expect(BARTG2P.fromBundle() != nil)
    }

    @Test("Known words match Python reference (greedy decoding is deterministic)")
    func referenceMatch() throws {
        let bart = try #require(BARTG2P.fromBundle())
        let expected: [(String, String)] = [
            ("hello", "hˈɛlO"),
            ("cat", "kˈæt"),
            ("world", "wˈɜɹld"),
        ]
        for (word, ref) in expected {
            let result = bart.predict(word)
            #expect(result == ref, "BART(\(word)): got \(result ?? "nil"), expected \(ref)")
        }
    }

    @Test("Edge cases: empty and too-long inputs")
    func edgeCases() throws {
        let bart = try #require(BARTG2P.fromBundle())
        #expect(bart.predict("") == nil)
        let tooLong = String(repeating: "a", count: 100)
        #expect(bart.predict(tooLong) == nil)
    }
}
