package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FatigueCuesTest {

    @Test
    fun canonicalGeometryMatchesExpectedRatios() {
        val points = MutableList(478) {
            Point2D(0.0, 0.0)
        }

        configureEye(
            points = points,
            eyeIndices = FatigueCues.RIGHT_EYE,
            irisIndices = FatigueCues.RIGHT_IRIS,
            xOffset = 0.0
        )

        configureEye(
            points = points,
            eyeIndices = FatigueCues.LEFT_EYE,
            irisIndices = FatigueCues.LEFT_IRIS,
            xOffset = 2.0
        )

        points[61] = Point2D(0.0, 2.0)
        points[291] = Point2D(1.0, 2.0)

        listOf(13 to 14, 82 to 87, 312 to 317)
            .forEach { (upper, lower) ->
                points[upper] = Point2D(0.5, 1.7)
                points[lower] = Point2D(0.5, 2.3)
            }

        val metrics = FatigueCues.fromPoints(points)

        assertTrue(metrics.valid)
        assertEquals(0.2, metrics.ear, 1e-9)
        assertEquals(0.6, metrics.mar, 1e-9)
        assertEquals(0.5, metrics.gaze.horizontal, 1e-9)
        assertEquals(0.5, metrics.gaze.vertical, 1e-9)
        assertEquals(
            0.0,
            metrics.gaze.horizontalAsymmetry,
            1e-9
        )
        assertEquals(
            0.0,
            metrics.gaze.verticalAsymmetry,
            1e-9
        )
    }

    @Test
    fun insufficientLandmarksAreInvalid() {
        val metrics = FatigueCues.fromPoints(
            List(477) { Point2D(0.0, 0.0) }
        )

        assertFalse(metrics.valid)
        assertTrue(metrics.ear.isNaN())
        assertTrue(metrics.mar.isNaN())
        assertTrue(metrics.gaze.horizontal.isNaN())
    }

    @Test
    fun collapsedGeometryIsInvalid() {
        val metrics = FatigueCues.fromPoints(
            List(478) { Point2D(0.5, 0.5) }
        )

        assertFalse(metrics.valid)
        assertTrue(metrics.ear.isNaN())
        assertTrue(metrics.mar.isNaN())
    }

    private fun configureEye(
        points: MutableList<Point2D>,
        eyeIndices: IntArray,
        irisIndices: IntArray,
        xOffset: Double
    ) {
        points[eyeIndices[0]] = Point2D(xOffset, 0.0)
        points[eyeIndices[1]] = Point2D(xOffset + 0.3, -0.1)
        points[eyeIndices[2]] = Point2D(xOffset + 0.7, -0.1)
        points[eyeIndices[3]] = Point2D(xOffset + 1.0, 0.0)
        points[eyeIndices[4]] = Point2D(xOffset + 0.7, 0.1)
        points[eyeIndices[5]] = Point2D(xOffset + 0.3, 0.1)

        irisIndices.forEach { index ->
            points[index] = Point2D(xOffset + 0.5, 0.0)
        }
    }
}
