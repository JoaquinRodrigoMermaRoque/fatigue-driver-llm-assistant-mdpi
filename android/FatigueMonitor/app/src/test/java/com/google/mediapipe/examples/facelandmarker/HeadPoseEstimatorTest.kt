package com.google.mediapipe.examples.facelandmarker

import kotlin.math.cos
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HeadPoseEstimatorTest {

    @Test
    fun identityMatrixProducesFrontalPose() {
        val matrix = matrixForPose(
            pitchDeg = 0.0,
            yawDeg = 0.0,
            rollDeg = 0.0,
            scale = 1.0
        )

        val pose =
            HeadPoseEstimator.fromColumnMajor(matrix)

        assertEquals(0.0, pose.pitchDeg, 1e-6)
        assertEquals(0.0, pose.yawDeg, 1e-6)
        assertEquals(0.0, pose.rollDeg, 1e-6)
        assertEquals(1.0, pose.scale, 1e-6)
        assertTrue(pose.fitError < 1e-6)
    }

    @Test
    fun recoversKnownAnglesAndUniformScale() {
        val matrix = matrixForPose(
            pitchDeg = -20.0,
            yawDeg = 30.0,
            rollDeg = 15.0,
            scale = 1.25
        )

        val pose =
            HeadPoseEstimator.fromColumnMajor(matrix)

        assertEquals(-20.0, pose.pitchDeg, 1e-4)
        assertEquals(30.0, pose.yawDeg, 1e-4)
        assertEquals(15.0, pose.rollDeg, 1e-4)
        assertEquals(1.25, pose.scale, 1e-5)
        assertTrue(pose.fitError < 1e-6)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsMatrixWithWrongSize() {
        HeadPoseEstimator.fromColumnMajor(
            FloatArray(15)
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteMatrix() {
        val matrix = matrixForPose(
            pitchDeg = 0.0,
            yawDeg = 0.0,
            rollDeg = 0.0,
            scale = 1.0
        )
        matrix[0] = Float.NaN

        HeadPoseEstimator.fromColumnMajor(matrix)
    }

    private fun matrixForPose(
        pitchDeg: Double,
        yawDeg: Double,
        rollDeg: Double,
        scale: Double
    ): FloatArray {
        val pitch = Math.toRadians(pitchDeg)
        val yaw = Math.toRadians(yawDeg)
        val roll = Math.toRadians(rollDeg)

        val cp = cos(pitch)
        val sp = sin(pitch)
        val cy = cos(yaw)
        val sy = sin(yaw)
        val cr = cos(roll)
        val sr = sin(roll)

        val rotation = arrayOf(
            doubleArrayOf(
                cr * cy,
                cr * sy * sp - sr * cp,
                cr * sy * cp + sr * sp
            ),
            doubleArrayOf(
                sr * cy,
                sr * sy * sp + cr * cp,
                sr * sy * cp - cr * sp
            ),
            doubleArrayOf(
                -sy,
                cy * sp,
                cy * cp
            )
        )

        val matrix = FloatArray(16)
        for (column in 0..2) {
            for (row in 0..2) {
                matrix[column * 4 + row] =
                    (
                        scale *
                            rotation[row][column]
                        ).toFloat()
            }
        }
        matrix[15] = 1.0f
        return matrix
    }
}
