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
    /// Normalized face bounding box [centerX, centerY, size] in [0, 1] frame coordinates.
    let head: [Double]?
    let t: Double?
}

/// One raw gaze reading stamped when it arrived, so calibration can discard
/// samples that still describe the previous dot (the video pipeline lags).
struct TimedGazeSample {
    let yawDegrees: Double
    let pitchDegrees: Double
    let headCenterX: Double
    let headCenterY: Double
    let headSize: Double
    let receivedAt: TimeInterval
}

struct GazeCalibrationCoefficients: Codable {
    static let featureCount = 9

    /// 9 coefficients for normalized horizontal screen coordinate u in [0, 1]:
    /// [1, yaw, pitch, yaw^2, yaw*pitch, pitch^2, headX, headY, headSize]
    let horizontalCoefficients: [Double]
    /// 9 coefficients for normalized vertical screen coordinate v in [0, 1].
    let verticalCoefficients: [Double]
    /// Average Euclidean error on the calibration points, expressed as a percentage of screen size.
    let averageErrorPercent: Double

    func predictNormalizedScreenPoint(
        yaw: Double,
        pitch: Double,
        headCenterX: Double,
        headCenterY: Double,
        headSize: Double
    ) -> CGPoint {
        let features = Self.featureVector(
            yaw: yaw,
            pitch: pitch,
            headCenterX: headCenterX,
            headCenterY: headCenterY,
            headSize: headSize
        )
        var predictedX = 0.0
        var predictedY = 0.0
        for index in 0..<Self.featureCount {
            predictedX += horizontalCoefficients[index] * features[index]
            predictedY += verticalCoefficients[index] * features[index]
        }
        return CGPoint(
            x: min(max(predictedX, 0.0), 1.0),
            y: min(max(predictedY, 0.0), 1.0)
        )
    }

    static func featureVector(
        yaw: Double,
        pitch: Double,
        headCenterX: Double,
        headCenterY: Double,
        headSize: Double
    ) -> [Double] {
        // Normalize gaze angles and center head-position terms around typical
        // framing so all features are on a comparable scale for ridge regression.
        let normalizedYaw = yaw / 45.0
        let normalizedPitch = pitch / 45.0
        let centeredHeadX = headCenterX - 0.5
        let centeredHeadY = headCenterY - 0.5
        let centeredHeadSize = headSize - 0.25
        return [
            1.0,
            normalizedYaw,
            normalizedPitch,
            normalizedYaw * normalizedYaw,
            normalizedYaw * normalizedPitch,
            normalizedPitch * normalizedPitch,
            centeredHeadX,
            centeredHeadY,
            centeredHeadSize,
        ]
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
    private static let calibrationDefaultsKey = "jarvisGazeCalibrationCoefficients_v4"

    /// One-Euro filter parameters: low minCutoff = rock-steady during fixations;
    /// higher beta = fast snap during saccades.
    private let oneEuroMinCutoff: Double = 0.6
    private let oneEuroBeta: Double = 4.0
    private let oneEuroDerivativeCutoff: Double = 1.0
    /// Ignore tiny jitter below 1.5% of screen size when fixating.
    private let fixationDeadzoneNormalized: Double = 0.015

    /// Time to wait after showing a calibration dot so the user's eyes finish
    /// moving and the (laggy) camera pipeline catches up with the new dot.
    private let calibrationSettleSeconds: Double = 1.6
    /// Time to collect gaze samples at each calibration dot.
    private let calibrationCollectSeconds: Double = 1.4
    /// Samples received right after collection starts can still describe the
    /// previous dot because video capture + inference + WebSocket add latency.
    /// Ignore this much of the collection window before computing the median.
    private let calibrationSampleWarmupSeconds: Double = 0.5

    // MARK: - Private Properties

    private var webSocketURLSession: URLSession?
    private let webSocketTrustDelegate = GazeWebSocketTrustDelegate()
    private var webSocketTask: URLSessionWebSocketTask?
    private var receiveLoopTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var calibrationTask: Task<Void, Never>?

    private var calibrationCoefficients: GazeCalibrationCoefficients?
    private var isCollectingSamplesForCurrentDot: Bool = false
    private var currentDotSamples: [TimedGazeSample] = []
    private var filterX = OneEuroFilter(minCutoff: 0.6, beta: 4.0, derivativeCutoff: 1.0)
    private var filterY = OneEuroFilter(minCutoff: 0.6, beta: 4.0, derivativeCutoff: 1.0)

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
        filterX.reset()
        filterY.reset()
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

        let headCenterX = (payload.head?.count ?? 0) >= 3 ? payload.head![0] : 0.5
        let headCenterY = (payload.head?.count ?? 0) >= 3 ? payload.head![1] : 0.5
        let headSize = (payload.head?.count ?? 0) >= 3 ? payload.head![2] : 0.25

        if isCalibrating && isCollectingSamplesForCurrentDot {
            currentDotSamples.append(
                TimedGazeSample(
                    yawDegrees: payload.yaw,
                    pitchDegrees: payload.pitch,
                    headCenterX: headCenterX,
                    headCenterY: headCenterY,
                    headSize: headSize,
                    receivedAt: Date().timeIntervalSince1970
                )
            )
        }

        let rawNormalizedPoint = mapAnglesToNormalizedScreenPoint(
            yaw: payload.yaw,
            pitch: payload.pitch,
            headCenterX: headCenterX,
            headCenterY: headCenterY,
            headSize: headSize
        )
        let timestamp = payload.t ?? Date().timeIntervalSince1970
        let filteredX = filterX.filter(value: Double(rawNormalizedPoint.x), timestamp: timestamp)
        let filteredY = filterY.filter(value: Double(rawNormalizedPoint.y), timestamp: timestamp)
        let candidatePoint = CGPoint(
            x: min(max(filteredX, 0.0), 1.0),
            y: min(max(filteredY, 0.0), 1.0)
        )

        if let previousPoint = smoothedNormalizedGazePoint {
            let distance = hypot(Double(candidatePoint.x - previousPoint.x), Double(candidatePoint.y - previousPoint.y))
            if distance >= fixationDeadzoneNormalized {
                smoothedNormalizedGazePoint = candidatePoint
            }
        } else {
            smoothedNormalizedGazePoint = candidatePoint
        }
    }

    private func mapAnglesToNormalizedScreenPoint(
        yaw: Double,
        pitch: Double,
        headCenterX: Double,
        headCenterY: Double,
        headSize: Double
    ) -> CGPoint {
        if let coefficients = calibrationCoefficients {
            return coefficients.predictNormalizedScreenPoint(
                yaw: yaw,
                pitch: pitch,
                headCenterX: headCenterX,
                headCenterY: headCenterY,
                headSize: headSize
            )
        }

        // Uncalibrated linear fallback so the gaze circle responds immediately.
        // L2CS-Net yaw spans roughly ±45°; pitch is negative when looking down.
        let normalizedX = min(max((yaw + 45.0) / 90.0, 0.0), 1.0)
        let normalizedY = min(max((-pitch - 5.0) / 55.0, 0.0), 1.0)
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
        var collectedPairs: [(target: CGPoint, yaw: Double, pitch: Double, headCenterX: Double, headCenterY: Double, headSize: Double)] = []

        for (stepIndex, targetPoint) in calibrationTargets.enumerated() {
            if Task.isCancelled { return }

            currentCalibrationStepIndex = stepIndex
            currentCalibrationTarget = targetPoint
            calibrationSampleProgress = 0.0
            isCollectingSamplesForCurrentDot = false
            currentDotSamples.removeAll()

            // Wait for the user's eyes to move to the new dot and for the
            // lagging camera pipeline to deliver gaze from that dot.
            let settleNanoseconds = UInt64(calibrationSettleSeconds * 1_000_000_000)
            try? await Task.sleep(nanoseconds: settleNanoseconds)
            if Task.isCancelled { return }

            // Collect gaze samples while animating the progress ring.
            let collectionStartTime = Date().timeIntervalSince1970
            isCollectingSamplesForCurrentDot = true
            let progressSteps = 20
            let stepSleepNanoseconds = UInt64((calibrationCollectSeconds / Double(progressSteps)) * 1_000_000_000)
            for tick in 1...progressSteps {
                try? await Task.sleep(nanoseconds: stepSleepNanoseconds)
                if Task.isCancelled { return }
                calibrationSampleProgress = Double(tick) / Double(progressSteps)
            }
            isCollectingSamplesForCurrentDot = false

            // Ignore warm-up samples that may still describe the previous dot.
            let warmedUpSamples = currentDotSamples.filter {
                $0.receivedAt >= collectionStartTime + calibrationSampleWarmupSeconds
            }
            let samplesForDot = warmedUpSamples.count >= 5 ? warmedUpSamples : currentDotSamples

            if !samplesForDot.isEmpty {
                let yawValues = samplesForDot.map(\.yawDegrees)
                let pitchValues = samplesForDot.map(\.pitchDegrees)
                let medianYaw = Self.robustMedian(of: yawValues)
                let medianPitch = Self.robustMedian(of: pitchValues)
                let medianHeadCenterX = Self.median(of: samplesForDot.map(\.headCenterX))
                let medianHeadCenterY = Self.median(of: samplesForDot.map(\.headCenterY))
                let medianHeadSize = Self.median(of: samplesForDot.map(\.headSize))
                let yawStandardDeviation = Self.standardDeviation(of: yawValues)
                let pitchStandardDeviation = Self.standardDeviation(of: pitchValues)
                print(
                    String(
                        format: "👁️ GazeTracking: dot %d/%d target=(%.2f, %.2f) -> yaw=%.2f°±%.2f pitch=%.2f°±%.2f head=(%.2f,%.2f,%.2f) (%d/%d samples)",
                        stepIndex + 1,
                        calibrationTargets.count,
                        targetPoint.x,
                        targetPoint.y,
                        medianYaw,
                        yawStandardDeviation,
                        medianPitch,
                        pitchStandardDeviation,
                        medianHeadCenterX,
                        medianHeadCenterY,
                        medianHeadSize,
                        samplesForDot.count,
                        currentDotSamples.count
                    )
                )
                collectedPairs.append((
                    target: targetPoint,
                    yaw: medianYaw,
                    pitch: medianPitch,
                    headCenterX: medianHeadCenterX,
                    headCenterY: medianHeadCenterY,
                    headSize: medianHeadSize
                ))
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
              saved.horizontalCoefficients.count == GazeCalibrationCoefficients.featureCount,
              saved.verticalCoefficients.count == GazeCalibrationCoefficients.featureCount else {
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

    /// Median after discarding samples that deviate more than 3 robust sigmas
    /// (from the median absolute deviation). Rejects blinks and tracking glitches.
    private static func robustMedian(of values: [Double]) -> Double {
        guard !values.isEmpty else { return 0.0 }
        let medianValue = median(of: values)
        let deviations = values.map { abs($0 - medianValue) }
        let medianAbsoluteDeviation = median(of: deviations)
        // 1.4826 * MAD approximates the standard deviation for normal data.
        // Floor it so a nearly constant signal does not reject every sample.
        let robustSigma = max(1.4826 * medianAbsoluteDeviation, 0.5)
        let inliers = values.filter { abs($0 - medianValue) <= 3.0 * robustSigma }
        return inliers.isEmpty ? medianValue : median(of: inliers)
    }

    private static func standardDeviation(of values: [Double]) -> Double {
        guard values.count > 1 else { return 0.0 }
        let mean = values.reduce(0.0, +) / Double(values.count)
        let sumOfSquaredDeviations = values.reduce(0.0) { partialResult, value in
            let deviation = value - mean
            return partialResult + deviation * deviation
        }
        return sqrt(sumOfSquaredDeviations / Double(values.count - 1))
    }

    /// Fits u = X * c_x and v = X * c_y using ridge regression: (X^T X + Lambda) c = X^T y.
    private static func fitQuadraticRidgeRegression(
        samples: [(target: CGPoint, yaw: Double, pitch: Double, headCenterX: Double, headCenterY: Double, headSize: Double)],
        lambda: Double
    ) -> GazeCalibrationCoefficients? {
        let featureCount = GazeCalibrationCoefficients.featureCount
        var normalMatrix = Array(repeating: Array(repeating: 0.0, count: featureCount), count: featureCount)
        var rhsX = Array(repeating: 0.0, count: featureCount)
        var rhsY = Array(repeating: 0.0, count: featureCount)

        for sample in samples {
            let features = GazeCalibrationCoefficients.featureVector(
                yaw: sample.yaw,
                pitch: sample.pitch,
                headCenterX: sample.headCenterX,
                headCenterY: sample.headCenterY,
                headSize: sample.headSize
            )
            for row in 0..<featureCount {
                rhsX[row] += features[row] * Double(sample.target.x)
                rhsY[row] += features[row] * Double(sample.target.y)
                for col in 0..<featureCount {
                    normalMatrix[row][col] += features[row] * features[col]
                }
            }
        }

        // Regularize non-intercept terms. Head-position terms (indices 6..8) get a
        // stronger penalty so sitting still during calibration does not overfit
        // bounding-box jitter, while deliberate head shifts are still learned.
        for index in 1..<featureCount {
            let penalty = index >= 6 ? max(lambda * 50.0, 0.05) : lambda
            normalMatrix[index][index] += penalty
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
            let predicted = candidate.predictNormalizedScreenPoint(
                yaw: sample.yaw,
                pitch: sample.pitch,
                headCenterX: sample.headCenterX,
                headCenterY: sample.headCenterY,
                headSize: sample.headSize
            )
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

/// Adaptive low-pass filter (Casiez et al., CHI 2012) that heavily smooths
/// jitter during fixations while remaining responsive during fast eye movements.
private struct OneEuroFilter {
    let minCutoff: Double
    let beta: Double
    let derivativeCutoff: Double

    private var previousTimestamp: Double?
    private var previousFilteredValue: Double?
    private var previousFilteredDerivative: Double = 0.0

    init(minCutoff: Double, beta: Double, derivativeCutoff: Double) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.derivativeCutoff = derivativeCutoff
    }

    mutating func reset() {
        previousTimestamp = nil
        previousFilteredValue = nil
        previousFilteredDerivative = 0.0
    }

    mutating func filter(value: Double, timestamp: Double) -> Double {
        guard let lastTimestamp = previousTimestamp,
              let lastValue = previousFilteredValue else {
            previousTimestamp = timestamp
            previousFilteredValue = value
            previousFilteredDerivative = 0.0
            return value
        }

        let dt = max(timestamp - lastTimestamp, 1.0 / 120.0)
        let rawDerivative = (value - lastValue) / dt
        let derivativeAlpha = Self.smoothingFactor(dt: dt, cutoff: derivativeCutoff)
        let filteredDerivative = previousFilteredDerivative + derivativeAlpha * (rawDerivative - previousFilteredDerivative)

        let adaptiveCutoff = minCutoff + beta * abs(filteredDerivative)
        let valueAlpha = Self.smoothingFactor(dt: dt, cutoff: adaptiveCutoff)
        let filteredValue = lastValue + valueAlpha * (value - lastValue)

        previousTimestamp = timestamp
        previousFilteredValue = filteredValue
        previousFilteredDerivative = filteredDerivative
        return filteredValue
    }

    private static func smoothingFactor(dt: Double, cutoff: Double) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }
}
