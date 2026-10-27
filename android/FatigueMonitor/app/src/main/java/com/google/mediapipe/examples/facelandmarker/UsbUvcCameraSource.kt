package com.google.mediapipe.examples.facelandmarker

import android.graphics.Bitmap
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import com.herohan.uvcapp.CameraHelper
import com.herohan.uvcapp.ICameraHelper
import com.serenegiant.usb.Size
import com.serenegiant.usb.UVCCamera
import java.nio.ByteBuffer
import kotlin.math.abs

/**
 * H7.13 direct UVC source.
 *
 * This class intentionally owns only the USB/UVC side of the pipeline.
 * Fatigue logic, MediaPipe, TTS and the local LLM remain outside this class.
 */
class UsbUvcCameraSource(
    private val listener: Listener,
    private val analysisIntervalMs: Long = 100L
) {

    interface Listener {
        fun onUvcStatus(status: String)
        fun onUvcOpened(device: UsbDevice, width: Int, height: Int)
        fun onUvcFrame(
            bitmap: Bitmap,
            frameTimeMs: Long,
            width: Int,
            height: Int
        )
        fun onUvcError(message: String, error: Throwable? = null)
    }

    companion object {
        private const val TAG = "H7_13_UVC"
        private const val TARGET_WIDTH = 640
        private const val TARGET_HEIGHT = 480
    }

    private var cameraHelper: ICameraHelper? = null
    private var previewSurface: Surface? = null
    private var selectedDeviceName: String? = null
    private var openedDeviceName: String? = null
    private var activeWidth = TARGET_WIDTH
    private var activeHeight = TARGET_HEIGHT
    private var lastAnalysisFrameMs = 0L
    private var deliveredFrameCount = 0L

    @Volatile
    private var started = false

    private val converter = Nv21BitmapConverter()

    private val stateCallback = object : ICameraHelper.StateCallback {
        override fun onAttach(device: UsbDevice) {
            if (!started || !isUvcDevice(device)) {
                return
            }
            Log.i(
                TAG,
                "ATTACH name=${device.deviceName} " +
                    "vid=${device.vendorId} pid=${device.productId}"
            )
            listener.onUvcStatus("Webcam USB detectada")

            if (selectedDeviceName == null) {
                selectDevice(device)
            }
        }

        override fun onDeviceOpen(
            device: UsbDevice,
            isFirstOpen: Boolean
        ) {
            if (!started) {
                return
            }
            openedDeviceName = device.deviceName
            Log.i(
                TAG,
                "DEVICE_OPEN name=${device.deviceName} " +
                    "first=$isFirstOpen"
            )
            listener.onUvcStatus("Permiso USB concedido; abriendo cámara")
            runCatching {
                cameraHelper?.openCamera()
            }.onFailure { error ->
                reportError("No se pudo abrir la webcam UVC", error)
            }
        }

        override fun onCameraOpen(device: UsbDevice) {
            if (!started) {
                return
            }

            val helper = cameraHelper ?: return
            try {
                // The Logitech stream is stable on this phone at VGA.
                // 1280x960 opens but does not actually deliver video frames
                // through this USB/UVC path, so select 640x480 before preview.
                val selectedSize = choosePreviewSize(
                    helper.supportedSizeList.orEmpty()
                )

                if (selectedSize != null) {
                    Log.i(
                        TAG,
                        "SELECT_SIZE $selectedSize"
                    )
                    helper.previewSize = selectedSize

                    // CameraHelper#getPreviewSize() may keep reporting the
                    // previous/default UVC size immediately after changing
                    // preview size. The callback payload is already VGA
                    // (640*480*1.5 = 460800 bytes), so the selected size is
                    // the authoritative size for frame decoding.
                    activeWidth = selectedSize.width
                    activeHeight = selectedSize.height
                } else {
                    helper.previewSize?.let { actualSize ->
                        activeWidth = actualSize.width
                        activeHeight = actualSize.height
                    }
                }

                helper.startPreview()

                previewSurface
                    ?.takeIf { it.isValid }
                    ?.let { surface ->
                        helper.addSurface(surface, false)
                    }

                helper.setFrameCallback(
                    { frame: ByteBuffer ->
                        handleFrame(frame)
                    },
                    UVCCamera.PIXEL_FORMAT_NV21
                )

                Log.i(
                    TAG,
                    "CAMERA_OPEN name=${device.deviceName} " +
                        "size=${activeWidth}x${activeHeight}"
                )
                listener.onUvcOpened(
                    device,
                    activeWidth,
                    activeHeight
                )
                listener.onUvcStatus(
                    "Webcam activa: ${activeWidth}x${activeHeight}"
                )
            } catch (error: Throwable) {
                reportError(
                    "La webcam fue detectada, pero no se pudo iniciar el video",
                    error
                )
            }
        }

        override fun onCameraClose(device: UsbDevice) {
            Log.i(TAG, "CAMERA_CLOSE name=${device.deviceName}")
            if (openedDeviceName == device.deviceName) {
                openedDeviceName = null
            }
        }

        override fun onDeviceClose(device: UsbDevice) {
            Log.i(TAG, "DEVICE_CLOSE name=${device.deviceName}")
            if (selectedDeviceName == device.deviceName) {
                selectedDeviceName = null
            }
            if (openedDeviceName == device.deviceName) {
                openedDeviceName = null
            }
        }

        override fun onDetach(device: UsbDevice) {
            Log.i(TAG, "DETACH name=${device.deviceName}")
            if (
                selectedDeviceName == device.deviceName ||
                openedDeviceName == device.deviceName
            ) {
                selectedDeviceName = null
                openedDeviceName = null
                listener.onUvcStatus("Webcam USB desconectada")
            }
        }

        override fun onCancel(device: UsbDevice) {
            Log.w(TAG, "PERMISSION_CANCEL name=${device.deviceName}")
            if (selectedDeviceName == device.deviceName) {
                selectedDeviceName = null
            }
            listener.onUvcStatus("Permiso USB cancelado")
        }
    }

    @Synchronized
    fun start(surface: Surface) {
        if (started) {
            if (surface != previewSurface) {
                updateSurface(surface)
            }
            return
        }

        if (!surface.isValid) {
            listener.onUvcStatus("Esperando superficie de la webcam")
            return
        }

        started = true
        previewSurface = surface
        selectedDeviceName = null
        openedDeviceName = null
        lastAnalysisFrameMs = 0L
        deliveredFrameCount = 0L

        val helper = CameraHelper()
        helper.setStateCallback(stateCallback)
        cameraHelper = helper

        val devices = helper.deviceList
            ?.filter(::isUvcDevice)
            .orEmpty()

        Log.i(
            TAG,
            "START uvcDevices=${devices.size}"
        )

        if (devices.isEmpty()) {
            listener.onUvcStatus(
                "No se encontró una webcam UVC conectada"
            )
            return
        }

        selectDevice(devices.first())
    }

    @Synchronized
    fun stop() {
        val helper = cameraHelper
        started = false
        selectedDeviceName = null
        openedDeviceName = null
        lastAnalysisFrameMs = 0L
        deliveredFrameCount = 0L

        previewSurface
            ?.takeIf { it.isValid }
            ?.let { surface ->
                runCatching {
                    helper?.removeSurface(surface)
                }
            }

        runCatching {
            helper?.closeCamera()
        }
        runCatching {
            helper?.release()
        }

        cameraHelper = null
        previewSurface = null
        listener.onUvcStatus("Webcam USB detenida")
        Log.i(TAG, "STOP")
    }

    fun release() {
        stop()
    }

    fun isRunning(): Boolean = started

    @Synchronized
    private fun updateSurface(surface: Surface) {
        val helper = cameraHelper
        previewSurface
            ?.takeIf { it.isValid }
            ?.let { old ->
                runCatching {
                    helper?.removeSurface(old)
                }
            }

        previewSurface = surface

        if (surface.isValid && openedDeviceName != null) {
            runCatching {
                helper?.addSurface(surface, false)
            }.onFailure { error ->
                reportError(
                    "No se pudo cambiar la superficie UVC",
                    error
                )
            }
        }
    }

    private fun selectDevice(device: UsbDevice) {
        if (!started) {
            return
        }

        selectedDeviceName = device.deviceName
        Log.i(
            TAG,
            "SELECT name=${device.deviceName} " +
                "vid=${device.vendorId} pid=${device.productId}"
        )
        listener.onUvcStatus("Solicitando acceso a la webcam USB")

        runCatching {
            cameraHelper?.selectDevice(device)
        }.onFailure { error ->
            selectedDeviceName = null
            reportError(
                "No se pudo solicitar acceso a la webcam USB",
                error
            )
        }
    }

    private fun handleFrame(frame: ByteBuffer) {
        if (!started) {
            return
        }

        val frameTimeMs = SystemClock.uptimeMillis()
        if (
            frameTimeMs - lastAnalysisFrameMs <
            analysisIntervalMs
        ) {
            return
        }
        lastAnalysisFrameMs = frameTimeMs

        val width = activeWidth
        val height = activeHeight
        val requiredBytes = width * height * 3 / 2
        val source = frame.duplicate()
        source.rewind()

        if (source.remaining() < requiredBytes) {
            Log.w(
                TAG,
                "FRAME_TOO_SMALL bytes=${source.remaining()} " +
                    "required=$requiredBytes " +
                    "size=${width}x${height}"
            )
            return
        }

        try {
            val nv21 = ByteArray(requiredBytes)
            source.get(nv21)
            val bitmap =
                converter.convert(nv21, width, height)

            deliveredFrameCount++
            if (
                deliveredFrameCount <= 3L ||
                deliveredFrameCount % 30L == 0L
            ) {
                Log.i(
                    TAG,
                    "FRAME_OK count=$deliveredFrameCount " +
                        "bytes=$requiredBytes " +
                        "size=${width}x${height} " +
                        "ts=$frameTimeMs"
                )
            }

            listener.onUvcFrame(
                bitmap = bitmap,
                frameTimeMs = frameTimeMs,
                width = width,
                height = height
            )
        } catch (error: Throwable) {
            reportError("Error procesando frame UVC", error)
        }
    }

    private fun choosePreviewSize(
        sizes: List<Size>
    ): Size? {
        if (sizes.isEmpty()) {
            Log.w(TAG, "No UVC preview sizes reported")
            return null
        }

        Log.i(
            TAG,
            "SUPPORTED_SIZES " +
                sizes.joinToString { "${it.width}x${it.height}" }
        )

        return sizes.firstOrNull {
            it.width == TARGET_WIDTH &&
                it.height == TARGET_HEIGHT
        } ?: sizes.minByOrNull { size ->
            val areaDistance = abs(
                size.width * size.height -
                    TARGET_WIDTH * TARGET_HEIGHT
            )
            val ratioDistance =
                abs(
                    size.width.toDouble() /
                        size.height.toDouble() -
                        4.0 / 3.0
                )
            areaDistance.toDouble() +
                ratioDistance * 100_000.0
        }
    }

    private fun isUvcDevice(device: UsbDevice): Boolean {
        return (0 until device.interfaceCount).any { index ->
            device.getInterface(index).interfaceClass ==
                UsbConstants.USB_CLASS_VIDEO
        }
    }

    private fun reportError(
        message: String,
        error: Throwable? = null
    ) {
        if (error == null) {
            Log.e(TAG, message)
        } else {
            Log.e(TAG, message, error)
        }
        listener.onUvcError(message, error)
    }

    /**
     * Pure Kotlin NV21 -> ARGB converter.
     * Avoids RenderScript and keeps H7.13 compatible with Android 12+.
     */
    private class Nv21BitmapConverter {
        private var argbBuffer = IntArray(0)

        fun convert(
            nv21: ByteArray,
            width: Int,
            height: Int
        ): Bitmap {
            val frameSize = width * height
            if (argbBuffer.size != frameSize) {
                argbBuffer = IntArray(frameSize)
            }

            var yp = 0
            for (j in 0 until height) {
                var uvp =
                    frameSize + (j shr 1) * width
                var u = 0
                var v = 0

                for (i in 0 until width) {
                    var y =
                        (nv21[yp].toInt() and 0xff) - 16
                    if (y < 0) {
                        y = 0
                    }

                    if ((i and 1) == 0) {
                        v =
                            (nv21[uvp++].toInt() and 0xff) -
                                128
                        u =
                            (nv21[uvp++].toInt() and 0xff) -
                                128
                    }

                    val y1192 = 1192 * y
                    var r = y1192 + 1634 * v
                    var g =
                        y1192 - 833 * v - 400 * u
                    var b = y1192 + 2066 * u

                    r = r.coerceIn(0, 262143)
                    g = g.coerceIn(0, 262143)
                    b = b.coerceIn(0, 262143)

                    argbBuffer[yp] =
                        -0x1000000 or
                            ((r shl 6) and 0xff0000) or
                            ((g shr 2) and 0xff00) or
                            ((b shr 10) and 0xff)
                    yp++
                }
            }

            return Bitmap.createBitmap(
                argbBuffer,
                width,
                height,
                Bitmap.Config.ARGB_8888
            )
        }
    }
}
