import Foundation

/// Errors thrown by the BART G2P model loader.
public enum BARTG2PError: Error, LocalizedError {
    /// Safetensors file is malformed or unreadable.
    case invalidSafetensors(String)
    /// A required tensor is missing from the safetensors file.
    case missingTensor(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSafetensors(let reason):
            "Invalid safetensors file: \(reason)"
        case .missingTensor(let name):
            "Missing tensor: \(name)"
        }
    }
}
