package com.google.mediapipe.examples.facelandmarker

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Punto bidimensional normalizado independiente de Android y MediaPipe.
 */
data class Point2D(
    val x: Double,
    val y: Double
)

/**
 * Posición normalizada del iris dentro de ambos ojos.
 */
data class GazeRatios(
    val horizontal: Double,
    val vertical: Double,
    val horizontalAsymmetry: Double,
    val verticalAsymmetry: Double
)

/**
 * Indicadores geométricos obtenidos para un frame.
 *
 * [valid] solo confirma que la geometría pudo calcularse. La calidad temporal
 * y los umbrales personalizados se incorporan en los siguientes hitos.
 */
data class FatigueCueMetrics(
    val ear: Double,
    val mar: Double,
    val gaze: GazeRatios,
    val valid: Boolean
)

/**
 * Portado directo de las funciones geométricas validadas en desktop/src/cues.py.
 */
object FatigueCues {
    private const val EPSILON = 1e-8
    private const val LANDMARK_COUNT = 478

    val RIGHT_EYE = intArrayOf(33, 160, 158, 133, 153, 144)
    val LEFT_EYE = intArrayOf(362, 385, 387, 263, 373, 380)

    val RIGHT_IRIS = intArrayOf(469, 470, 471, 472)
    val LEFT_IRIS = intArrayOf(474, 475, 476, 477)

    val MOUTH_CORNERS = intArrayOf(61, 291)
    val MOUTH_VERTICAL = arrayOf(
        intArrayOf(13, 14),
        intArrayOf(82, 87),
        intArrayOf(312, 317)
    )

    private interface PointSource {
        val size: Int
        fun point(index: Int): Point2D
    }

    private class MediaPipePointSource(
        private val landmarks: List<NormalizedLandmark>
    ) : PointSource {
        override val size: Int
            get() = landmarks.size

        override fun point(index: Int): Point2D {
            val landmark = landmarks[index]
            return Point2D(
                x = landmark.x().toDouble(),
                y = landmark.y().toDouble()
            )
        }
    }

    private class ListPointSource(
        private val points: List<Point2D>
    ) : PointSource {
        override val size: Int
            get() = points.size

        override fun point(index: Int): Point2D = points[index]
    }

    fun fromLandmarks(
        landmarks: List<NormalizedLandmark>
    ): FatigueCueMetrics {
        return calculate(MediaPipePointSource(landmarks))
    }

    /**
     * Entrada independiente de MediaPipe utilizada por las pruebas unitarias.
     */
    fun fromPoints(
        points: List<Point2D>
    ): FatigueCueMetrics {
        return calculate(ListPointSource(points))
    }

    private fun calculate(
        source: PointSource
    ): FatigueCueMetrics {
        if (source.size < LANDMARK_COUNT) {
            return invalidMetrics()
        }

        val ear = meanEar(source)
        val mar = mouthAspectRatio(source)
        val gaze = meanGazeRatios(source)

        val valid = listOf(
            ear,
            mar,
            gaze.horizontal,
            gaze.vertical,
            gaze.horizontalAsymmetry,
            gaze.verticalAsymmetry
        ).all { it.isFinite() }

        return FatigueCueMetrics(
            ear = ear,
            mar = mar,
            gaze = gaze,
            valid = valid
        )
    }

    private fun distance(
        first: Point2D,
        second: Point2D
    ): Double {
        return hypot(
            first.x - second.x,
            first.y - second.y
        )
    }

    private fun clamp(
        value: Double,
        minimum: Double = 0.0,
        maximum: Double = 1.0
    ): Double {
        return max(minimum, min(maximum, value))
    }

    private fun eyeAspectRatio(
        source: PointSource,
        indices: IntArray
    ): Double {
        require(indices.size == 6) {
            "indices debe contener seis landmarks."
        }

        val p1 = source.point(indices[0])
        val p2 = source.point(indices[1])
        val p3 = source.point(indices[2])
        val p4 = source.point(indices[3])
        val p5 = source.point(indices[4])
        val p6 = source.point(indices[5])

        val denominator = 2.0 * distance(p1, p4)

        if (!denominator.isFinite() || denominator < EPSILON) {
            return Double.NaN
        }

        val numerator =
            distance(p2, p6) +
                distance(p3, p5)

        return numerator / denominator
    }

    private fun meanEar(
        source: PointSource
    ): Double {
        val values = listOf(
            eyeAspectRatio(source, RIGHT_EYE),
            eyeAspectRatio(source, LEFT_EYE)
        ).filter { it.isFinite() }

        return if (values.isEmpty()) {
            Double.NaN
        } else {
            values.average()
        }
    }

    private fun mouthAspectRatio(
        source: PointSource
    ): Double {
        val leftCorner = source.point(MOUTH_CORNERS[0])
        val rightCorner = source.point(MOUTH_CORNERS[1])
        val horizontalDistance = distance(leftCorner, rightCorner)

        if (
            !horizontalDistance.isFinite() ||
            horizontalDistance < EPSILON
        ) {
            return Double.NaN
        }

        val verticalMean = MOUTH_VERTICAL
            .map { pair ->
                distance(
                    source.point(pair[0]),
                    source.point(pair[1])
                )
            }
            .average()

        return verticalMean / horizontalDistance
    }

    private fun irisCenter(
        source: PointSource,
        indices: IntArray
    ): Point2D {
        if (indices.isEmpty()) {
            return Point2D(Double.NaN, Double.NaN)
        }

        return Point2D(
            x = indices.map { source.point(it).x }.average(),
            y = indices.map { source.point(it).y }.average()
        )
    }

    private fun projectOnAxis(
        point: Point2D,
        origin: Point2D,
        axisX: Double,
        axisY: Double
    ): Double {
        return (
            (point.x - origin.x) * axisX +
                (point.y - origin.y) * axisY
            )
    }

    private fun eyeGazeRatios(
        source: PointSource,
        eyeIndices: IntArray,
        irisIndices: IntArray
    ): Point2D {
        require(eyeIndices.size == 6) {
            "eyeIndices debe contener seis landmarks."
        }

        val cornerA = source.point(eyeIndices[0])
        val cornerB = source.point(eyeIndices[3])

        val imageLeft: Point2D
        val imageRight: Point2D

        if (cornerA.x <= cornerB.x) {
            imageLeft = cornerA
            imageRight = cornerB
        } else {
            imageLeft = cornerB
            imageRight = cornerA
        }

        val eyeAxisX = imageRight.x - imageLeft.x
        val eyeAxisY = imageRight.y - imageLeft.y
        val eyeWidth = hypot(eyeAxisX, eyeAxisY)

        if (!eyeWidth.isFinite() || eyeWidth < EPSILON) {
            return Point2D(Double.NaN, Double.NaN)
        }

        val horizontalAxisX = eyeAxisX / eyeWidth
        val horizontalAxisY = eyeAxisY / eyeWidth

        var verticalAxisX = -horizontalAxisY
        var verticalAxisY = horizontalAxisX

        if (verticalAxisY < 0.0) {
            verticalAxisX *= -1.0
            verticalAxisY *= -1.0
        }

        val iris = irisCenter(source, irisIndices)

        val irisHorizontal = projectOnAxis(
            iris,
            imageLeft,
            horizontalAxisX,
            horizontalAxisY
        )

        val horizontalRatio = irisHorizontal / eyeWidth

        val upperPosition = listOf(
            source.point(eyeIndices[1]),
            source.point(eyeIndices[2])
        ).map { point ->
            projectOnAxis(
                point,
                imageLeft,
                verticalAxisX,
                verticalAxisY
            )
        }.average()

        val lowerPosition = listOf(
            source.point(eyeIndices[4]),
            source.point(eyeIndices[5])
        ).map { point ->
            projectOnAxis(
                point,
                imageLeft,
                verticalAxisX,
                verticalAxisY
            )
        }.average()

        val topPosition = min(upperPosition, lowerPosition)
        val bottomPosition = max(upperPosition, lowerPosition)
        val eyeHeight = bottomPosition - topPosition

        if (!eyeHeight.isFinite() || eyeHeight < EPSILON) {
            return Point2D(Double.NaN, Double.NaN)
        }

        val irisVertical = projectOnAxis(
            iris,
            imageLeft,
            verticalAxisX,
            verticalAxisY
        )

        val verticalRatio =
            (irisVertical - topPosition) / eyeHeight

        return Point2D(
            x = clamp(horizontalRatio),
            y = clamp(verticalRatio)
        )
    }

    private fun meanGazeRatios(
        source: PointSource
    ): GazeRatios {
        val right = eyeGazeRatios(
            source,
            RIGHT_EYE,
            RIGHT_IRIS
        )

        val left = eyeGazeRatios(
            source,
            LEFT_EYE,
            LEFT_IRIS
        )

        val values = listOf(
            right.x,
            right.y,
            left.x,
            left.y
        )

        if (!values.all { it.isFinite() }) {
            return invalidGaze()
        }

        return GazeRatios(
            horizontal = (right.x + left.x) / 2.0,
            vertical = (right.y + left.y) / 2.0,
            horizontalAsymmetry = abs(right.x - left.x),
            verticalAsymmetry = abs(right.y - left.y)
        )
    }

    private fun invalidGaze(): GazeRatios {
        return GazeRatios(
            horizontal = Double.NaN,
            vertical = Double.NaN,
            horizontalAsymmetry = Double.NaN,
            verticalAsymmetry = Double.NaN
        )
    }

    private fun invalidMetrics(): FatigueCueMetrics {
        return FatigueCueMetrics(
            ear = Double.NaN,
            mar = Double.NaN,
            gaze = invalidGaze(),
            valid = false
        )
    }
}
