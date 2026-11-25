//
//  GazeTrackingManager.swift
//  leanring-buddy
//
//  Connects to the Jarvis GazeFormer WebSocket (/gaze-live/gaze),
//  maps (yaw, pitch) angles to normalized screen coordinates via a
//  9-point 2D quadratic ridge-regression calibration, and publishes
//  smoothed gaze coordinates for the overlay circle.
//

import Combine
import Foundation
import SwiftUI

struct GazeWebSocketPayload: Decodable {
    let face: Bool
    let yaw: Double
    let pitch: Double
    let gaze: [Double]?
    let t: Double?
}

struct GazeCalibrationCoefficients: Codable {
    /// 6 polynomial coefficients for normalized horizontal screen coordinate u in [0, 1]:
    /// u = c0 + c1*yaw + c2*pitch + c3*yaw^2 + c4*yaw*pitch + c5*pitch^2
    let horizontalCoefficients: [Double]
    /// 6 polynomial coefficients for normalized vertical screen coordinate v in [0, 1].
    let verticalCoefficients: [Double]
    /// Average Euclidean error on the calibration points, expressed as a percentage of screen size.
    let averageErrorPercent: Double

    func predictNormalizedScreenPoint(yaw: Double, pitch: Double) -> CGPoint {
        let features = Self.polynomialFeatures(yaw: yaw, pitch: pitch)
        var predictedX = 0.0
        var predictedY = 0.0
        for index in 0..<6 {
            predictedX += horizontalCoefficients[index] * features[index]
            predictedY += verticalCoefficients[index] * features[index]
        }
        return CGPoint(
            x: min(max(predictedX, 0.0), 1.0),
            y: min(max(predictedY, 0.0), 1.0)
        )
    }

    static func polynomialFeatures(yaw: Double, pitch: Double) -> [Double] {
        return [1.0, yaw, pitch, yaw * yaw, yaw * pitch, pitch * pitch]
    }
}

@MainActor
final class GazeTrackingManager: ObservableObject {

    // MARK: - Published State

    @Published var isEnabled: Bool = false
    @Published private(set) var isConnected: Bool = false
    @Published private(set) var hasFace: Bool = false
    @Published private(set) var latestYawDegrees: Double = 0.0
    @Published private(set) var latestPitchDegrees: Double = 0.0

    /// Smoothed normalized gaze point in [0, 1] x [0, 1] (top-left origin).
    @Published private(set) var smoothedNormalizedGazePoint: CGPoint?

    // MARK: - Calibration State

    @Published private(set) var isCalibrating: Bool = false
    @Published private(set) var currentCalibrationTarget: CGPoint?
    @Published private(set) var currentCalibrationStepIndex: Int = 0
    @Published private(set) var calibrationSampleProgress: Double = 0.0
    @Published private(set) var isCalibrated: Bool = false
    @Published private(set) var lastCalibrationErrorPercent: Double?

    /// 3x3 calibration grid inset 8% from the screen edges.
    let calibrationTargets: [CGPoint] = [
        CGPoint(x: 0.08, y: 0.08),
        CGPoint(x: 0.50, y: 0.08),
        CGPoint(x: 0.92, y: 0.08),
        CGPoint(x: 0.08, y: 0.50),
        CGPoint(x: 0.50, y: 0.50),
        CGPoint(x: 0.92, y: 0.50),
        CGPoint(x: 0.08, y: 0.92),
        CGPoint(x: 0.50, y: 0.92),
        CGPoint(x: 0.92, y: 0.92),
    ]

    // MARK: - Configuration

    private static let defaultWebSocketURLString = "wss://192.168.0.226:8443/gaze-live/gaze"
    private static let webSocketURLDefaultsKey = "jarvisGazeWebSocketURL"
    private static let calibrationDefaultsKey = "jarvisGazeCalibrationCoefficients"

    /// Exponential moving average smoothing factor (0 = frozen, 1 = raw/unsmoothed).
    private let smoothingAlpha: Double = 0.25

    /// Time to wait after showing a calibration dot so the user's eyes finish moving to it.
    private let calibrationSettleSeconds: Double = 0.45
    /// Time to collect gaze samples at each calibration dot.
    private let calibrationCollectSeconds: Double = 0.85

    // MARK: - Private Properties

    private var webSocketURLSession: URLSession?
    private let webSocketTrustDelegate = GazeWebSocketTrustDelegate()
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveLoopTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var calibrationTask: Task<Void, Never>?

    private var calibrationCoefficients: GazeCalibrationCoefficients?
    private var isCollectingSamplesForCurrentDot: Bool = false
    private var currentDotYawSamples: [Double] = []
    private var currentDotPitchSamples: [Double] = []

    var webSocketURLString: String {
        get {
            UserDefaults.standard.string(forKey: Self.webSocketURLDefaultsKey) ?? Self.defaultWebSocketURLString
        }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.webSocketURLDefaultsKey)
        }
    }

    // MARK: - Initialization

    init() {
        loadSavedCalibration()
    }

    // MARK: - Lifecycle

    func setEnabled(_ enabled: Bool) {
        isEnabled = enabled
        if enabled {
            start()
        } else {
            cancelCalibration()
            stop()
        }
    }

    func start() {
        guard webSocketTask == nil else { return }
        connectWebSocket()
    }

    func stop() {
        reconnectTask?.cancel()
        reconnectTask = nil
        receiveLoopTask?.cancel()
        receiveLoopTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        isConnected = false
        hasFace = false
        smoothedNormalizedGazePoint = nil
    }

    // MARK: - WebSocket Connection

    private func connectWebSocket() {
        guard let url = URL(string: webSocketURLString) else {
            print("⚠️ GazeTracking: invalid WebSocket URL \(webSocketURLString)")
            return
        }

        if webSocketURLSession == nil {
            let configuration = URLSessionConfiguration.default
            configuration.waitsForConnectivity = false
            webSocketURLSession = URLSession(
                configuration: configuration,
                delegate: webSocketTrustDelegate,
                delegateQueue: nil
            )
        }

        guard let session = webSocketURLSession else { return }
        let task = session.webSocketTask(with: url)
        webSocketTask = task
        task.resume()
        isConnected = true
        print("👁️ GazeTracking: connecting to \(webSocketURLString)")

        receiveLoopTask?.cancel()
        receiveLoopTask = Task { [weak self] in
            await self?.runReceiveLoop(for: task)
        }
    }

    private func runReceiveLoop(for task: URLSessionWebSocketTask) async {
        while !Task.isCancelled && isEnabled {
            do {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    handleIncomingJSONText(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        handleIncomingJSONText(text)
                    }
                @unknown default:
                    break
                }
            } catch {
                if Task.isCancelled || !isEnabled { return }
                print("⚠️ GazeTracking: WebSocket disconnected (\(error.localizedDescription))")
                break
            }
        }

        isConnected = false
        hasFace = false
        webSocketTask = nil

        if isEnabled && !Task.isCancelled {
            scheduleReconnect()
        }
    }

    private func scheduleReconnect() {
        reconnectTask?.cancel()
        reconnectTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard let self, !Task.isCancelled, self.isEnabled else { return }
            self.connectWebSocket()
        }
    }

    private func handleIncomingJSONText(_ text: String) {
        guard let data = text.data(using: .utf8),
              let payload = try? JSONDecoder().decode(GazeWebSocketPayload.self, from: data) else {
            return
        }

        hasFace = payload.face
        guard payload.face else { return }

        latestYawDegrees = payload.yaw
        latestPitchDegrees = payload.pitch

        if isCalibrating && isCollectingSamplesForCurrentDot {
            currentDotYawSamples.append(payload.yaw)
            currentDotPitchSamples.append(payload.pitch)
        }

        let rawNormalizedPoint = mapAnglesToNormalizedScreenPoint(yaw: payload.yaw, pitch: payload.pitch)
        if let previousPoint = smoothedNormalizedGazePoint {
            let smoothedX = previousPoint.x + smoothingAlpha * (rawNormalizedPoint.x - previousPoint.x)
            let smoothedY = previousPoint.y + smoothingAlpha * (rawNormalizedPoint.y - previousPoint.y)
            smoothedNormalizedGazePoint = CGPoint(x: smoothedX, y: smoothedY)
        } else {
            smoothedNormalizedGazePoint = rawNormalizedPoint
        }
    }

    private func mapAnglesToNormalizedScreenPoint(yaw: Double, pitch: Double) -> CGPoint {
        if let coefficients = calibrationCoefficients {
            return coefficients.predictNormalizedScreenPoint(yaw: yaw, pitch: pitch)
        }

        // Uncalibrated linear fallback so the gaze circle responds immediately.
        let normalizedX = min(max((yaw + 25.0) / 50.0, 0.0), 1.0)
        let normalizedY = min(max((pitch + 15.0) / 30.0, 0.0), 1.0)
        return CGPoint(x: normalizedX, y: normalizedY)
    }

    // MARK: - 9-Point Calibration

    func startCalibration() {
        if !isEnabled {
            setEnabled(true)
        }

        calibrationTask?.cancel()
        isCalibrating = true
        currentCalibrationStepIndex = 0
        calibrationSampleProgress = 0.0
        currentCalibrationTarget = calibrationTargets.first

        calibrationTask = Task { [weak self] in
            await self?.runCalibrationSequence()
        }
    }

    func cancelCalibration() {
        calibrationTask?.cancel()
        calibrationTask = nil
        isCalibrating = false
        isCollectingSamplesForCurrentDot = false
        currentCalibrationTarget = nil
        calibrationSampleProgress = 0.0
    }

    func resetCalibration() {
        calibrationCoefficients = nil
        isCalibrated = false
        lastCalibrationErrorPercent = nil
        UserDefaults.standard.removeObject(forKey: Self.calibrationDefaultsKey)
    }

    private func runCalibrationSequence() async {
        var collectedPairs: [(target: CGPoint, yaw: Double, pitch: Double)] = []

        for (stepIndex, targetPoint) in calibrationTargets.enumerated() {
            if Task.isCancelled { return }

            currentCalibrationStepIndex = stepIndex
            currentCalibrationTarget = targetPoint
            calibrationSampleProgress = 0.0
            isCollectingSamplesForCurrentDot = false
            currentDotYawSamples.removeAll()
            currentDotPitchSamples.removeAll()

            // Wait for the user's eyes to settle on the new dot.
            let settleNanoseconds = UInt64(calibrationSettleSeconds * 1_000_000_000)
            try? await Task.sleep(nanoseconds: settleNanoseconds)
            if Task.isCancelled { return }

            // Collect gaze samples while animating the progress ring.
            isCollectingSamplesForCurrentDot = true
            let progressSteps = 20
            let stepSleepNanoseconds = UInt64((calibrationCollectSeconds / Double(progressSteps)) * 1_000_000_000)
            for tick in 1...progressSteps {
                try? await Task.sleep(nanoseconds: stepSleepNanoseconds)
                if Task.isCancelled { return }
                calibrationSampleProgress = Double(tick) / Double(progressSteps)
            }
            isCollectingSamplesForCurrentDot = false

            if !currentDotYawSamples.isEmpty && !currentDotPitchSamples.isEmpty {
                let medianYaw = Self.median(of: currentDotYawSamples)
                let medianPitch = Self.median(of: currentDotPitchSamples)
                collectedPairs.append((target: targetPoint, yaw: medianYaw, pitch: medianPitch))
            }
        }

        isCalibrating = false
        currentCalibrationTarget = nil
        calibrationSampleProgress = 0.0

        guard collectedPairs.count >= 6 else {
            print("⚠️ GazeTracking: calibration failed — only \(collectedPairs.count) valid points collected")
            return
        }

        if let fitted = Self.fitQuadraticRidgeRegression(samples: collectedPairs, lambda: 1e-3) {
            calibrationCoefficients = fitted
            isCalibrated = true
            lastCalibrationErrorPercent = fitted.averageErrorPercent
            saveCalibration(fitted)
            print("👁️ GazeTracking: calibration complete (avg error \(String(format: "%.1f", fitted.averageErrorPercent))%)")
        } else {
            print("⚠️ GazeTracking: failed to solve calibration linear system")
        }
    }

    // MARK: - Persistence

    private func loadSavedCalibration() {
        guard let data = UserDefaults.standard.data(forKey: Self.calibrationDefaultsKey),
              let saved = try? JSONDecoder().decode(GazeCalibrationCoefficients.self, from: data),
              saved.horizontalCoefficients.count == 6,
              saved.verticalCoefficients.count == 6 else {
            return
        }
        calibrationCoefficients = saved
        isCalibrated = true
        lastCalibrationErrorPercent = saved.averageErrorPercent
    }

    private func saveCalibration(_ coefficients: GazeCalibrationCoefficients) {
        guard let data = try? JSONEncoder().encode(coefficients) else { return }
        UserDefaults.standard.set(data, forKey: Self.calibrationDefaultsKey)
    }

    // MARK: - Math Helpers

    private static func median(of values: [Double]) -> Double {
        let sortedValues = values.sorted()
        let count = sortedValues.count
        if count % 2 == 1 {
            return sortedValues[count / 2]
        }
        return 0.5 * (sortedValues[count / 2 - 1] + sortedValues[count / 2])
    }

    /// Fits u = X * c_x and v = X * c_y using ridge regression: (X^T X + lambda * I) c = X^T y.
    private static func fitQuadraticRidgeRegression(
        samples: [(target: CGPoint, yaw: Double, pitch: Double)],
        lambda: Double
    ) -> GazeCalibrationCoefficients? {
        let featureCount = 6
        var normalMatrix = Array(repeating: Array(repeating: 0.0, count: featureCount), count: featureCount)
        var rhsX = Array(repeating: 0.0, count: featureCount)
        var rhsY = Array(repeating: 0.0, count: featureCount)

        for sample in samples {
            let features = GazeCalibrationCoefficients.polynomialFeatures(yaw: sample.yaw, pitch: sample.pitch)
            for row in 0..<featureCount {
                rhsX[row] += features[row] * Double(sample.target.x)
                rhsY[row] += features[row] * Double(sample.target.y)
                for col in 0..<featureCount {
                    normalMatrix[row][col] += features[row] * features[col]
                }
            }
        }

        // Regularize non-intercept terms to keep the 6x6 system well-conditioned.
        for index in 1..<featureCount {
            normalMatrix[index][index] += lambda
        }

        guard let horizontalCoefficients = solveLinearSystem(matrix: normalMatrix, rhs: rhsX),
              let verticalCoefficients = solveLinearSystem(matrix: normalMatrix, rhs: rhsY) else {
            return nil
        }

        let candidate = GazeCalibrationCoefficients(
            horizontalCoefficients: horizontalCoefficients,
            verticalCoefficients: verticalCoefficients,
            averageErrorPercent: 0.0
        )

        var totalError = 0.0
        for sample in samples {
            let predicted = candidate.predictNormalizedScreenPoint(yaw: sample.yaw, pitch: sample.pitch)
            let dx = Double(predicted.x - sample.target.x)
            let dy = Double(predicted.y - sample.target.y)
            totalError += sqrt(dx * dx + dy * dy)
        }
        let averageErrorPercent = (totalError / Double(samples.count)) * 100.0

        return GazeCalibrationCoefficients(
            horizontalCoefficients: horizontalCoefficients,
            verticalCoefficients: verticalCoefficients,
            averageErrorPercent: averageErrorPercent
        )
    }

    /// Gaussian elimination with partial pivoting for small dense square systems.
    private static func solveLinearSystem(matrix: [[Double]], rhs: [Double]) -> [Double]? {
        let size = rhs.count
        var augmented = matrix
        var vector = rhs

        for pivotCol in 0..<size {
            var maxRow = pivotCol
            var maxAbsValue = abs(augmented[pivotCol][pivotCol])
            for candidateRow in (pivotCol + 1)..<size {
                let candidateValue = abs(augmented[candidateRow][pivotCol])
                if candidateValue > maxAbsValue {
                    maxAbsValue = candidateValue
                    maxRow = candidateRow
                }
            }

            guard maxAbsValue > 1e-12 else { return nil }

            if maxRow != pivotCol {
                augmented.swapAt(pivotCol, maxRow)
                vector.swapAt(pivotCol, maxRow)
            }

            let pivotValue = augmented[pivotCol][pivotCol]
            for row in (pivotCol + 1)..<size {
                let factor = augmented[row][pivotCol] / pivotValue
                vector[row] -= factor * vector[pivotCol]
                for col in pivotCol..<size {
                    augmented[row][col] -= factor * augmented[pivotCol][col]
                }
            }
        }

        var solution = Array(repeating: 0.0, count: size)
        for row in stride(from: size - 1, through: 0, by: -1) {
            var sum = vector[row]
            for col in (row + 1)..<size {
                sum -= augmented[row][col] * solution[col]
            }
            solution[row] = sum / augmented[row][row]
        }
        return solution
    }
}

/// Trusts Caddy's local LAN certificate for the Jarvis gaze server so
/// URLSessionWebSocketTask doesn't reject IP-based internal TLS certs.
private final class GazeWebSocketTrustDelegate: NSObject, URLSessionDelegate {
    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
           let serverTrust = challenge.protectionSpace.serverTrust {
            completionHandler(.useCredential, URLCredential(trust: serverTrust))
            return
        }
        completionHandler(.performDefaultHandling, nil)
    }
}
