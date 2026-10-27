/*
 * Head-pose extraction from MediaPipe's 4x4 facial transformation matrix.
 */
package com.google.mediapipe.examples.facelandmarker

import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sqrt

data class HeadPose(
    val pitchDeg: Double,
    val yawDeg: Double,
    val rollDeg: Double,
    val scale: Double,
    val fitError: Double
)

object HeadPoseEstimator {
    private const val EPSILON = 1e-12
    private const val RADIANS_TO_DEGREES =
        180.0 / Math.PI

    fun fromColumnMajor(
        transformation: FloatArray
    ): HeadPose {
        require(transformation.size == 16) {
            "La matriz facial debe contener 16 valores."
        }
        require(
            transformation.all {
                it.toDouble().isFinite()
            }
        ) {
            "La matriz facial contiene valores no finitos."
        }

        // MediaPipe Java devuelve un arreglo 4x4 column-major.
        val column0 = Vec3(
            transformation[0].toDouble(),
            transformation[1].toDouble(),
            transformation[2].toDouble()
        )
        val column1 = Vec3(
            transformation[4].toDouble(),
            transformation[5].toDouble(),
            transformation[6].toDouble()
        )
        val column2 = Vec3(
            transformation[8].toDouble(),
            transformation[9].toDouble(),
            transformation[10].toDouble()
        )

        val norm0 = column0.norm()
        val norm1 = column1.norm()
        val norm2 = column2.norm()

        require(
            norm0 > EPSILON &&
                norm1 > EPSILON &&
                norm2 > EPSILON
        ) {
            "La matriz facial tiene escala degenerada."
        }

        val scale = (norm0 + norm1 + norm2) / 3.0

        // La especificación de Face Landmarker garantiza escala
        // uniforme y rotación. Gram-Schmidt elimina el pequeño error
        // numérico sin añadir una dependencia de álgebra lineal.
        val xAxis = column0.normalized()
        var yAxis = (
            column1 -
                xAxis * column1.dot(xAxis)
            ).normalized()
        var zAxis = xAxis.cross(yAxis).normalized()

        if (zAxis.dot(column2) < 0.0) {
            yAxis *= -1.0
            zAxis *= -1.0
        }

        // Recalcular Y conserva una base ortonormal de determinante +1.
        yAxis = zAxis.cross(xAxis).normalized()

        val rotation = arrayOf(
            doubleArrayOf(xAxis.x, yAxis.x, zAxis.x),
            doubleArrayOf(xAxis.y, yAxis.y, zAxis.y),
            doubleArrayOf(xAxis.z, yAxis.z, zAxis.z)
        )

        val linear = arrayOf(
            doubleArrayOf(
                column0.x,
                column1.x,
                column2.x
            ),
            doubleArrayOf(
                column0.y,
                column1.y,
                column2.y
            ),
            doubleArrayOf(
                column0.z,
                column1.z,
                column2.z
            )
        )

        var errorSquared = 0.0
        var linearSquared = 0.0

        for (row in 0..2) {
            for (column in 0..2) {
                val expected =
                    scale * rotation[row][column]
                val difference =
                    linear[row][column] - expected
                errorSquared += difference * difference
                linearSquared +=
                    linear[row][column] *
                        linear[row][column]
            }
        }

        val fitError =
            sqrt(errorSquared) /
                maxOf(sqrt(linearSquared), EPSILON)

        val sinYaw =
            (-rotation[2][0]).coerceIn(-1.0, 1.0)
        val yaw = asin(sinYaw)
        val cosYaw = cos(yaw)

        val pitch: Double
        val roll: Double

        if (kotlin.math.abs(cosYaw) > 1e-6) {
            pitch = atan2(
                rotation[2][1],
                rotation[2][2]
            )
            roll = atan2(
                rotation[1][0],
                rotation[0][0]
            )
        } else {
            pitch = atan2(
                -rotation[1][2],
                rotation[1][1]
            )
            roll = 0.0
        }

        return HeadPose(
            pitchDeg = pitch * RADIANS_TO_DEGREES,
            yawDeg = yaw * RADIANS_TO_DEGREES,
            rollDeg = roll * RADIANS_TO_DEGREES,
            scale = scale,
            fitError = fitError
        )
    }

    private data class Vec3(
        val x: Double,
        val y: Double,
        val z: Double
    ) {
        operator fun minus(other: Vec3): Vec3 {
            return Vec3(
                x - other.x,
                y - other.y,
                z - other.z
            )
        }

        operator fun times(value: Double): Vec3 {
            return Vec3(
                x * value,
                y * value,
                z * value
            )
        }

        fun dot(other: Vec3): Double {
            return (
                x * other.x +
                    y * other.y +
                    z * other.z
                )
        }

        fun cross(other: Vec3): Vec3 {
            return Vec3(
                y * other.z - z * other.y,
                z * other.x - x * other.z,
                x * other.y - y * other.x
            )
        }

        fun norm(): Double {
            return sqrt(dot(this))
        }

        fun normalized(): Vec3 {
            val length = norm()
            require(length > EPSILON) {
                "No se puede normalizar un vector degenerado."
            }
            return this * (1.0 / length)
        }
    }
}
