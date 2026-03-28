// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "swift-bart-g2p",
    platforms: [.macOS(.v14), .iOS(.v18)],
    products: [
        .library(name: "BARTG2P", targets: ["BARTG2P"]),
    ],
    targets: [
        .target(
            name: "BARTG2P",
            path: "Sources/BARTG2P",
            resources: [
                .process("Resources"),
            ]
        ),
        .executableTarget(
            name: "GenerateRerankerData",
            dependencies: ["BARTG2P"],
            path: "Sources/GenerateRerankerData"
        ),
        .testTarget(
            name: "BARTG2PTests",
            dependencies: ["BARTG2P"],
            path: "Tests/BARTG2PTests",
            resources: [
                .copy("cmudict_ref_1000.tsv"),
            ]
        ),
    ]
)
