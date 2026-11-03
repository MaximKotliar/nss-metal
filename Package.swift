// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NSSMetal",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "NSSMetal", targets: ["NSSMetal"]),
        .executable(name: "nss-bench", targets: ["nss-bench"]),
    ],
    targets: [
        .target(
            name: "NSSMetal",
            path: "Sources/NSSMetal",
            resources: [.copy("Shaders")],   // nss.metal: compiled at setup when no nss.metallib is given
            publicHeadersPath: "include",
            linkerSettings: [.linkedFramework("Metal"), .linkedFramework("QuartzCore")]
        ),
        .executableTarget(
            name: "nss-bench",
            dependencies: ["NSSMetal"],
            path: "Sources/nss-bench",
            linkerSettings: [.linkedFramework("MetalFX"), .linkedFramework("ImageIO"), .linkedFramework("CoreGraphics"),
                             .linkedFramework("UniformTypeIdentifiers")]
        ),
    ]
)
