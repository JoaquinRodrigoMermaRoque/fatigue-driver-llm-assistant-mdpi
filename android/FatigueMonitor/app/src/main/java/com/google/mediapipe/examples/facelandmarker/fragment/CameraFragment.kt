/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.facelandmarker.fragment

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.text.InputType
import android.util.Log
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.viewpager2.widget.ViewPager2.SCROLL_STATE_DRAGGING
import androidx.viewpager2.widget.ViewPager2.SCROLL_STATE_IDLE
import androidx.viewpager2.widget.ViewPager2.SCROLL_STATE_SETTLING
import androidx.viewpager2.widget.ViewPager2.ScrollState
import com.google.mediapipe.examples.facelandmarker.AblationMode
import com.google.mediapipe.examples.facelandmarker.AndroidOfflineAlertSpeaker
import com.google.mediapipe.examples.facelandmarker.AsyncLocalLlmEngine
import com.google.mediapipe.examples.facelandmarker.AlertUrgency
import com.google.mediapipe.examples.facelandmarker.ContextualAlertPlanner
import com.google.mediapipe.examples.facelandmarker.ContextualAlertDecision
import com.google.mediapipe.examples.facelandmarker.CalibrationCameraSource
import com.google.mediapipe.examples.facelandmarker.DriverCalibrationProfile
import com.google.mediapipe.examples.facelandmarker.DriverCalibrationStore
import com.google.mediapipe.examples.facelandmarker.DriverContextEntry
import com.google.mediapipe.examples.facelandmarker.FaceLandmarkerHelper
import com.google.mediapipe.examples.facelandmarker.FatigueAlertSnapshot
import com.google.mediapipe.examples.facelandmarker.FatigueAlertSoundPlayer
import com.google.mediapipe.examples.facelandmarker.FatigueCues
import com.google.mediapipe.examples.facelandmarker.FatigueEngine
import com.google.mediapipe.examples.facelandmarker.FatigueEventRecorder
import com.google.mediapipe.examples.facelandmarker.FatigueHistoryFrame
import com.google.mediapipe.examples.facelandmarker.FatigueCueMetrics
import com.google.mediapipe.examples.facelandmarker.H8AblationFeatureLogger
import com.google.mediapipe.examples.facelandmarker.H8AblationFrame
import com.google.mediapipe.examples.facelandmarker.FatigueState
import com.google.mediapipe.examples.facelandmarker.EyeTemporalAnalyzer
import com.google.mediapipe.examples.facelandmarker.GazeTemporalAnalyzer
import com.google.mediapipe.examples.facelandmarker.HeadPose
import com.google.mediapipe.examples.facelandmarker.HeadPoseEstimator
import com.google.mediapipe.examples.facelandmarker.HeadPoseTemporalAnalyzer
import com.google.mediapipe.examples.facelandmarker.MouthTemporalAnalyzer
import com.google.mediapipe.examples.facelandmarker.NeutralCalibrationSession
import com.google.mediapipe.examples.facelandmarker.MainViewModel
import com.google.mediapipe.examples.facelandmarker.MediaPipePerformanceTelemetry
import com.google.mediapipe.examples.facelandmarker.R
import com.google.mediapipe.examples.facelandmarker.RecentActivityLevel
import com.google.mediapipe.examples.facelandmarker.SQLiteDriverContextStore
import com.google.mediapipe.examples.facelandmarker.SQLiteFatigueHistoryStore
import com.google.mediapipe.examples.facelandmarker.UsbUvcCameraSource
import com.google.mediapipe.examples.facelandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.io.File
import java.util.Calendar
import java.util.Locale
import java.util.Optional
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

class CameraFragment : Fragment(), FaceLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Face Landmarker"
        private const val HISTORY_TAG = "FatigueHistory"
        private const val ALERT_CONTEXT_TAG =
            "ContextualAlert"
        private const val H8_2_ABLATION_TAG =
            "H8_2_ABLATION"
        private const val EXTRA_ABLATION_MODE =
            "ablation_mode"
        private const val H8_3_EXPERIMENT_TAG =
            "H8_3_EXPERIMENT"
        private const val EXTRA_H8_LOG_FEATURES =
            "h8_log_features"
        private const val EXTRA_H8_RUN_ID =
            "h8_run_id"
        private const val EXTRA_H8_SUBJECT_ID =
            "h8_subject_id"
        private const val EXTRA_H8_CLIP_ID =
            "h8_clip_id"
        private const val EXTRA_H8_SOURCE =
            "h8_source"
        private const val EXTRA_CONTEXT_SESSION_START =
            "context_session_started_elapsed_ms"
        private const val EXTRA_CONTEXT_PROMPT_SHOWN =
            "context_prompt_shown"

        // El dispositivo tarda aproximadamente 70-107 ms por inferencia.
        // Diez FPS evita acumular conversiones RGBA sin perder los eventos
        // faciales relevantes para el monitor de fatiga.
        private const val ANALYSIS_INTERVAL_MS = 100L
        private const val MOBILE_EAR_THRESHOLD = 0.14
        private const val MOBILE_MAR_THRESHOLD = 0.27
        private const val MOBILE_GAZE_LOW = 0.37
        private const val MOBILE_GAZE_HIGH = 0.62
        private const val MAX_GAZE_ASYMMETRY = 0.20
    }

    private enum class CameraInputMode {
        INTERNAL,
        USB
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private val faceBlendshapesResultAdapter by lazy {
        FaceBlendshapesResultAdapter()
    }

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT
    private var cameraInputMode = CameraInputMode.INTERNAL
    private var usbSurface: Surface? = null
    private lateinit var usbCameraSource: UsbUvcCameraSource
    private val usbFrameInFlight = AtomicBoolean(false)
    private var lastAnalysisTimeMs = 0L
    private val temporalLock = Any()
    private val eyeTemporalAnalyzer = EyeTemporalAnalyzer(
        windowS = 15.0,
        earThreshold = MOBILE_EAR_THRESHOLD
    )
    private val mouthTemporalAnalyzer = MouthTemporalAnalyzer(
        marThreshold = MOBILE_MAR_THRESHOLD
    )
    private val gazeTemporalAnalyzer = GazeTemporalAnalyzer(
        horizontalLow = MOBILE_GAZE_LOW,
        horizontalHigh = MOBILE_GAZE_HIGH,
        verticalEnabled = false
    )
    private val poseTemporalAnalyzer =
        HeadPoseTemporalAnalyzer()
    private val requestedAblationModeCode by lazy {
        requireActivity().intent
            .getStringExtra(EXTRA_ABLATION_MODE)
    }
    private val ablationMode by lazy {
        AblationMode.fromCode(
            requestedAblationModeCode
        )
    }
    private val fatigueEngine by lazy {
        FatigueEngine(
            ablationMode = ablationMode
        )
    }
    private val contextualAlertPlanner =
        ContextualAlertPlanner()
    private val performanceTelemetry =
        MediaPipePerformanceTelemetry(windowSize = 30)
    private val acknowledgementRequested =
        AtomicBoolean(false)
    private lateinit var alertSpeaker:
        AndroidOfflineAlertSpeaker
    private lateinit var alertSoundPlayer:
        FatigueAlertSoundPlayer
    private lateinit var localLlmEngine:
        AsyncLocalLlmEngine
    private var llmClosurePriorityCancelArmed = true
    private var deferredClosureLlmDecision:
        ContextualAlertDecision? = null
    private lateinit var historyStore:
        SQLiteFatigueHistoryStore
    private lateinit var driverContextStore:
        SQLiteDriverContextStore
    private var contextualSessionStartedElapsedMs = 0L
    private var driverContextPromptShownThisSession = false
    private lateinit var eventRecorder:
        FatigueEventRecorder
    private lateinit var calibrationStore:
        DriverCalibrationStore
    private var h8FeatureLogger:
        H8AblationFeatureLogger? = null
    private var h8RequiredSource:
        CalibrationCameraSource? = null
    private var h8FirstFrameLogged = false
    private val calibrationLock = Any()
    private var neutralCalibrationSession:
        NeutralCalibrationSession? = null
    private var calibrationApplyLogCount = 0L

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        resetTemporalAnalyzers()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the FaceLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if (faceLandmarkerHelper.isClose()) {
                faceLandmarkerHelper.setupFaceLandmarker()
            }
        }

        if (
            cameraInputMode == CameraInputMode.USB &&
            this::usbCameraSource.isInitialized
        ) {
            usbSurface
                ?.takeIf { it.isValid }
                ?.let { surface ->
                    startUsbCamera(surface)
                }
        }
    }

    override fun onPause() {
        super.onPause()
        resetTemporalAnalyzers()
        if (this::usbCameraSource.isInitialized) {
            usbCameraSource.stop()
        }
        usbFrameInFlight.set(false)
        if (this::alertSpeaker.isInitialized) {
            alertSpeaker.stop()
        }
        if (this::alertSoundPlayer.isInitialized) {
            alertSoundPlayer.stop()
        }
        if (this::localLlmEngine.isInitialized) {
            localLlmEngine.cancelGeneration(
                reason = "fragment_onPause"
            )
        }
        if(this::faceLandmarkerHelper.isInitialized) {
            viewModel.setMaxFaces(faceLandmarkerHelper.maxNumFaces)
            viewModel.setMinFaceDetectionConfidence(faceLandmarkerHelper.minFaceDetectionConfidence)
            viewModel.setMinFaceTrackingConfidence(faceLandmarkerHelper.minFaceTrackingConfidence)
            viewModel.setMinFacePresenceConfidence(faceLandmarkerHelper.minFacePresenceConfidence)
            viewModel.setDelegate(faceLandmarkerHelper.currentDelegate)

            // Close the FaceLandmarkerHelper and release resources
            backgroundExecutor.execute { faceLandmarkerHelper.clearFaceLandmarker() }
        }
    }

    override fun onDestroyView() {
        if (this::usbCameraSource.isInitialized) {
            usbCameraSource.release()
        }
        usbSurface = null
        usbFrameInFlight.set(false)

        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )

        if (this::alertSpeaker.isInitialized) {
            alertSpeaker.shutdown()
        }

        if (this::alertSoundPlayer.isInitialized) {
            alertSoundPlayer.release()
        }

        if (this::localLlmEngine.isInitialized) {
            localLlmEngine.close()
        }

        if (this::historyStore.isInitialized) {
            historyStore.close()
        }

        if (this::driverContextStore.isInitialized) {
            driverContextStore.close()
        }

        h8FeatureLogger?.let { logger ->
            runCatching {
                logger.close()
            }.onFailure { error ->
                Log.e(
                    H8_3_EXPERIMENT_TAG,
                    "CLOSE_FAILED run_id=" +
                        logger.featureRunId,
                    error
                )
            }
            Log.i(
                H8_3_EXPERIMENT_TAG,
                "STOP run_id=" +
                    logger.featureRunId +
                    " rows=" +
                    logger.rowsWritten() +
                    " path=" +
                    logger.file.absolutePath
            )
        }
        h8FeatureLogger = null
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)
        alertSpeaker = AndroidOfflineAlertSpeaker(
            requireContext().applicationContext
        )
        alertSoundPlayer = FatigueAlertSoundPlayer()
        localLlmEngine = AsyncLocalLlmEngine(
            requireContext().applicationContext
        )
        localLlmEngine.initialize()

        historyStore = SQLiteFatigueHistoryStore(
            requireContext().applicationContext
        )
        driverContextStore = SQLiteDriverContextStore(
            requireContext().applicationContext
        )
        val activityIntent = requireActivity().intent
        val persistedSessionStart =
            activityIntent.getLongExtra(
                EXTRA_CONTEXT_SESSION_START,
                0L
            )
        contextualSessionStartedElapsedMs =
            if (persistedSessionStart > 0L) {
                persistedSessionStart
            } else {
                SystemClock.elapsedRealtime().also {
                    activityIntent.putExtra(
                        EXTRA_CONTEXT_SESSION_START,
                        it
                    )
                }
            }
        driverContextPromptShownThisSession =
            activityIntent.getBooleanExtra(
                EXTRA_CONTEXT_PROMPT_SHOWN,
                false
            )
        eventRecorder = FatigueEventRecorder(
            repository = historyStore
        )
        calibrationStore = DriverCalibrationStore(
            requireContext().applicationContext
        )
        initializeH8FeatureLogger()
        Log.i(
            H8_2_ABLATION_TAG,
            "SELECTED requested=" +
                (requestedAblationModeCode ?: "<none>") +
                " mode=" +
                ablationMode.code +
                " cues=" +
                ablationMode.enabledCues.joinToString(",")
        )
        Log.i(
            HISTORY_TAG,
            "Historial local disponible: " +
                historyStore.count() +
                " eventos previos"
        )

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        with(fragmentCameraBinding.recyclerviewResults) {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = faceBlendshapesResultAdapter
        }

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        usbCameraSource = UsbUvcCameraSource(
            listener = object : UsbUvcCameraSource.Listener {
                override fun onUvcStatus(status: String) {
                    Log.i("H7_13_UVC", status)
                }

                override fun onUvcOpened(
                    device: android.hardware.usb.UsbDevice,
                    width: Int,
                    height: Int
                ) {
                    Log.i(
                        "H7_13_UVC",
                        "READY vid=${device.vendorId} " +
                            "pid=${device.productId} " +
                            "size=${width}x${height}"
                    )
                    activity?.runOnUiThread {
                        if (
                            _fragmentCameraBinding != null &&
                            isAdded &&
                            cameraInputMode ==
                                CameraInputMode.USB
                        ) {
                            Toast.makeText(
                                requireContext(),
                                "Webcam USB activa: " +
                                    "${width}x${height}",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                }

                override fun onUvcFrame(
                    bitmap: Bitmap,
                    frameTimeMs: Long,
                    width: Int,
                    height: Int
                ) {
                    if (
                        cameraInputMode !=
                            CameraInputMode.USB ||
                        !usbFrameInFlight.compareAndSet(
                            false,
                            true
                        )
                    ) {
                        bitmap.recycle()
                        return
                    }

                    backgroundExecutor.execute {
                        try {
                            if (
                                this@CameraFragment::faceLandmarkerHelper.isInitialized &&
                                !faceLandmarkerHelper.isClose()
                            ) {
                                faceLandmarkerHelper
                                    .detectLiveStreamBitmap(
                                        bitmap = bitmap,
                                        frameTime =
                                            frameTimeMs,
                                        rotationDegrees = 0,
                                        mirrorHorizontally =
                                            false
                                    )
                            } else {
                                bitmap.recycle()
                            }
                        } catch (error: Throwable) {
                            Log.e(
                                "H7_13_UVC",
                                "USB frame analysis failed",
                                error
                            )
                            onError(
                                error.message
                                    ?: "USB frame analysis failed"
                            )
                        } finally {
                            usbFrameInFlight.set(false)
                        }
                    }
                }

                override fun onUvcError(
                    message: String,
                    error: Throwable?
                ) {
                    Log.e(
                        "H7_13_UVC",
                        message,
                        error
                    )
                    activity?.runOnUiThread {
                        if (
                            _fragmentCameraBinding != null &&
                            isAdded
                        ) {
                            Toast.makeText(
                                requireContext(),
                                message,
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                }
            },
            analysisIntervalMs = ANALYSIS_INTERVAL_MS
        )

        fragmentCameraBinding.usbViewFinder.holder
            .addCallback(
                object : SurfaceHolder.Callback {
                    override fun surfaceCreated(
                        holder: SurfaceHolder
                    ) {
                        usbSurface = holder.surface
                        if (
                            cameraInputMode ==
                                CameraInputMode.USB
                        ) {
                            startUsbCamera(
                                holder.surface
                            )
                        }
                    }

                    override fun surfaceChanged(
                        holder: SurfaceHolder,
                        format: Int,
                        width: Int,
                        height: Int
                    ) {
                        usbSurface = holder.surface
                    }

                    override fun surfaceDestroyed(
                        holder: SurfaceHolder
                    ) {
                        if (
                            cameraInputMode ==
                                CameraInputMode.USB
                        ) {
                            usbCameraSource.stop()
                        }
                        usbSurface = null
                    }
                }
            )

        fragmentCameraBinding.cameraSourceToggle
            .setOnClickListener {
                switchCameraInput()
            }

        fragmentCameraBinding.calibrationButton
            .setOnClickListener {
                startNeutralCalibration()
            }

        fragmentCameraBinding.driverContextButton
            .setOnClickListener {
                markDriverContextPromptShown()
                showDriverContextDialog()
            }

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

        // Create the FaceLandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            faceLandmarkerHelper = FaceLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minFaceDetectionConfidence = viewModel.currentMinFaceDetectionConfidence,
                minFaceTrackingConfidence = viewModel.currentMinFaceTrackingConfidence,
                minFacePresenceConfidence = viewModel.currentMinFacePresenceConfidence,
                maxNumFaces = viewModel.currentMaxFaces,
                currentDelegate = viewModel.currentDelegate,
                faceLandmarkerHelperListener = this
            )
            performanceTelemetry.reset(
                faceLandmarkerHelper.activeDelegate
            )
        }

        // Attach listeners to UI control widgets
        initBottomSheetControls()

        fragmentCameraBinding.fatigueMetrics
            .setOnClickListener {
                acknowledgementRequested.set(true)
                if (this::alertSpeaker.isInitialized) {
                    alertSpeaker.stop()
                }
                if (this::localLlmEngine.isInitialized) {
                    localLlmEngine.cancelGeneration(
                        reason = "alert_acknowledged"
                    )
                }
                Toast.makeText(
                    requireContext(),
                    R.string.fatigue_ack_requested,
                    Toast.LENGTH_SHORT
                ).show()
            }

        logCalibrationProfile(calibrationSource())

        fragmentCameraBinding.root.postDelayed(
            {
                if (
                    isAdded &&
                    _fragmentCameraBinding != null
                ) {
                    showDriverContextIntroIfNeeded()
                }
            },
            700L
        )
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings
        fragmentCameraBinding.bottomSheetLayout.maxFacesValue.text =
            viewModel.currentMaxFaces.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinFaceDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinFaceTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinFacePresenceConfidence
            )

        // When clicked, lower face detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceDetectionConfidence >= 0.2) {
                faceLandmarkerHelper.minFaceDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise face detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceDetectionConfidence <= 0.8) {
                faceLandmarkerHelper.minFaceDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower face tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceTrackingConfidence >= 0.2) {
                faceLandmarkerHelper.minFaceTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise face tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceTrackingConfidence <= 0.8) {
                faceLandmarkerHelper.minFaceTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower face presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (faceLandmarkerHelper.minFacePresenceConfidence >= 0.2) {
                faceLandmarkerHelper.minFacePresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise face presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (faceLandmarkerHelper.minFacePresenceConfidence <= 0.8) {
                faceLandmarkerHelper.minFacePresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, reduce the number of faces that can be detected at a
        // time
        fragmentCameraBinding.bottomSheetLayout.maxFacesMinus.setOnClickListener {
            if (faceLandmarkerHelper.maxNumFaces > 1) {
                faceLandmarkerHelper.maxNumFaces--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of faces that can be detected
        // at a time
        fragmentCameraBinding.bottomSheetLayout.maxFacesPlus.setOnClickListener {
            if (faceLandmarkerHelper.maxNumFaces < 2) {
                faceLandmarkerHelper.maxNumFaces++
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        faceLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "FaceLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset Facelandmarker
    // helper.
    private fun updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxFacesValue.text =
            faceLandmarkerHelper.maxNumFaces.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                faceLandmarkerHelper.minFaceDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                faceLandmarkerHelper.minFaceTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                faceLandmarkerHelper.minFacePresenceConfidence
            )

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        backgroundExecutor.execute {
            faceLandmarkerHelper.clearFaceLandmarker()
            faceLandmarkerHelper.setupFaceLandmarker()
            performanceTelemetry.reset(
                faceLandmarkerHelper.activeDelegate
            )
        }
        fragmentCameraBinding.overlay.clear()
        resetTemporalAnalyzers()
    }

    private fun initializeH8FeatureLogger() {
        val intent = requireActivity().intent
        if (
            !intent.getBooleanExtra(
                EXTRA_H8_LOG_FEATURES,
                false
            )
        ) {
            return
        }

        val externalRoot =
            requireContext().getExternalFilesDir(null)
                ?: requireContext().filesDir
        val outputDir = File(
            externalRoot,
            "h8_experiments"
        )
        val subjectId =
            intent.getStringExtra(EXTRA_H8_SUBJECT_ID)
                ?.takeIf { it.isNotBlank() }
                ?: "pilot_001"
        val clipId =
            intent.getStringExtra(EXTRA_H8_CLIP_ID)
                ?.takeIf { it.isNotBlank() }
                ?: "unlabeled_clip"
        val requestedRunId =
            intent.getStringExtra(EXTRA_H8_RUN_ID)
        h8RequiredSource =
            intent.getStringExtra(EXTRA_H8_SOURCE)
                ?.trim()
                ?.uppercase(Locale.US)
                ?.let { raw ->
                    runCatching {
                        CalibrationCameraSource
                            .valueOf(raw)
                    }.getOrNull()
                }
        h8FirstFrameLogged = false

        h8FeatureLogger = runCatching {
            H8AblationFeatureLogger(
                outputDir = outputDir,
                subjectId = subjectId,
                clipId = clipId,
                requestedRunId = requestedRunId,
                flushEvery = 1
            )
        }.onFailure { error ->
            Log.e(
                H8_3_EXPERIMENT_TAG,
                "INIT_FAILED requested_run_id=" +
                    (requestedRunId ?: "<auto>") +
                    " output_dir=" +
                    outputDir.absolutePath,
                error
            )
        }.getOrNull()

        h8FeatureLogger?.let { logger ->
            Log.i(
                H8_3_EXPERIMENT_TAG,
                "START run_id=" +
                    logger.featureRunId +
                    " subject_id=" +
                    subjectId +
                    " clip_id=" +
                    clipId +
                    " mode=" +
                    ablationMode.code +
                    " required_source=" +
                    (h8RequiredSource?.name ?: "ANY") +
                    " path=" +
                    logger.file.absolutePath
            )
        }
    }

    private fun writeH8ExperimentFrame(
        frame: H8AblationFrame
    ) {
        val logger = h8FeatureLogger ?: return
        val requiredSource = h8RequiredSource

        if (
            requiredSource != null &&
            frame.source != requiredSource
        ) {
            return
        }

        runCatching {
            logger.write(frame)
            if (!h8FirstFrameLogged) {
                h8FirstFrameLogged = true
                Log.i(
                    H8_3_EXPERIMENT_TAG,
                    "CAPTURE_STARTED run_id=" +
                        logger.featureRunId +
                        " source=" +
                        frame.source.name
                )
            }
        }.onFailure { error ->
            Log.e(
                H8_3_EXPERIMENT_TAG,
                "WRITE_FAILED run_id=" +
                    logger.featureRunId +
                    " rows=" +
                    logger.rowsWritten(),
                error
            )
            runCatching {
                logger.close()
            }
            h8FeatureLogger = null
        }
    }

    private fun resetTemporalAnalyzers() {
        synchronized(temporalLock) {
            eyeTemporalAnalyzer.reset()
            mouthTemporalAnalyzer.reset()
            gazeTemporalAnalyzer.reset()
            poseTemporalAnalyzer.reset()
            fatigueEngine.reset()
            contextualAlertPlanner
                .resetSessionTracking()
            acknowledgementRequested.set(false)
            if (this::eventRecorder.isInitialized) {
                eventRecorder.resetSessionTracking()
            }
        }
        calibrationApplyLogCount = 0L
    }

    private fun calibrationSource(): CalibrationCameraSource {
        return when (cameraInputMode) {
            CameraInputMode.INTERNAL ->
                CalibrationCameraSource.INTERNAL
            CameraInputMode.USB ->
                CalibrationCameraSource.USB
        }
    }

    private fun startNeutralCalibration() {
        val wasRunning = synchronized(calibrationLock) {
            if (neutralCalibrationSession != null) {
                neutralCalibrationSession = null
                true
            } else {
                false
            }
        }

        if (wasRunning) {
            Log.i("H7_14_CAL", "CANCELLED")
            Toast.makeText(
                requireContext(),
                "Calibración cancelada",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val source = calibrationSource()
        val sourceLabel =
            if (source == CalibrationCameraSource.USB) {
                "webcam USB"
            } else {
                "cámara del teléfono"
            }

        AlertDialog.Builder(requireContext())
            .setTitle("Calibración individual")
            .setMessage(
                "Mantén el dispositivo en su posición final. " +
                    "Mira normalmente hacia la carretera, no hacia la cámara.\n\n" +
                    "Se capturarán 3 segundos de referencia con la " +
                    sourceLabel +
                    ". Esta calibración ajustará la referencia de pose " +
                    "y mirada; no modifica los umbrales de fatiga."
            )
            .setNegativeButton("Cancelar", null)
            .setPositiveButton("Iniciar") { _, _ ->
                synchronized(calibrationLock) {
                    neutralCalibrationSession =
                        NeutralCalibrationSession(source)
                }
                Log.i(
                    "H7_14_CAL",
                    "START source=${source.name}"
                )
                Toast.makeText(
                    requireContext(),
                    "Mira al frente durante 3 segundos",
                    Toast.LENGTH_LONG
                ).show()
            }
            .show()
    }

    private fun consumeNeutralCalibrationSample(
        timestampMs: Long,
        cueMetrics:
            com.google.mediapipe.examples.facelandmarker.FatigueCueMetrics?,
        headPose:
            com.google.mediapipe.examples.facelandmarker.HeadPose?
    ) {
        val completedProfile =
            synchronized(calibrationLock) {
                val session =
                    neutralCalibrationSession
                        ?: return

                if (session.source != calibrationSource()) {
                    Log.w(
                        "H7_14_CAL",
                        "CANCEL_SOURCE_CHANGED from=${session.source.name} " +
                            "to=${calibrationSource().name}"
                    )
                    neutralCalibrationSession = null
                    return
                }

                val before = session.sampleCount()
                val profile = session.add(
                    timestampMs = timestampMs,
                    cues = cueMetrics,
                    pose = headPose
                )
                val count = session.sampleCount()

                if (
                    count != before &&
                    (
                        count == 1 ||
                        count % 10 == 0
                    )
                ) {
                    Log.i(
                        "H7_14_CAL",
                        "SAMPLE source=${session.source.name} " +
                            "count=$count"
                    )
                }

                if (profile != null) {
                    neutralCalibrationSession = null
                }

                profile
            }

        if (completedProfile == null) {
            return
        }

        calibrationStore.save(completedProfile)

        // Drop only the pose/gaze smoothing state so the new baseline is
        // applied from the next valid frame without mixing old coordinates.
        synchronized(temporalLock) {
            gazeTemporalAnalyzer.reset()
            poseTemporalAnalyzer.reset()
        }
        calibrationApplyLogCount = 0L

        Log.i(
            "H7_14_CAL",
            String.format(
                Locale.US,
                "COMPLETE source=%s samples=%d yaw=%.2f pitch=%.2f roll=%.2f gaze=%.3f/%.3f ear=%.4f mar=%.4f scale=%.4f",
                completedProfile.source.name,
                completedProfile.sampleCount,
                completedProfile.yawBaseDeg,
                completedProfile.pitchBaseDeg,
                completedProfile.rollBaseDeg,
                completedProfile.gazeCenterH,
                completedProfile.gazeCenterV,
                completedProfile.earOpen,
                completedProfile.marRest,
                completedProfile.faceScaleBase
            )
        )

        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null && isAdded) {
                Toast.makeText(
                    requireContext(),
                    String.format(
                        Locale.US,
                        "Calibración %s guardada: Y/P/R %.1f / %.1f / %.1f°",
                        completedProfile.source.name,
                        completedProfile.yawBaseDeg,
                        completedProfile.pitchBaseDeg,
                        completedProfile.rollBaseDeg
                    ),
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    private fun logCalibrationProfile(
        source: CalibrationCameraSource
    ) {
        if (!this::calibrationStore.isInitialized) {
            return
        }

        val profile = calibrationStore.load(source)
        if (profile == null) {
            Log.i(
                "H7_14_CAL",
                "NO_PROFILE source=${source.name}"
            )
            return
        }

        Log.i(
            "H7_14_CAL",
            String.format(
                Locale.US,
                "PROFILE source=%s samples=%d yaw=%.2f pitch=%.2f roll=%.2f gaze=%.3f/%.3f ear=%.4f mar=%.4f scale=%.4f",
                profile.source.name,
                profile.sampleCount,
                profile.yawBaseDeg,
                profile.pitchBaseDeg,
                profile.rollBaseDeg,
                profile.gazeCenterH,
                profile.gazeCenterV,
                profile.earOpen,
                profile.marRest,
                profile.faceScaleBase
            )
        )
    }

    private fun activeCalibrationProfile():
        DriverCalibrationProfile? {
        if (!this::calibrationStore.isInitialized) {
            return null
        }

        return calibrationStore.load(
            calibrationSource()
        )
    }

    private fun applyPoseCalibration(
        headPose: HeadPose?,
        profile: DriverCalibrationProfile?
    ): HeadPose? {
        if (headPose == null || profile == null) {
            return headPose
        }

        // HeadPoseTemporalAnalyzer subtracts its own canonical baseline.
        // Shift the measured pose so the user's calibrated driving posture
        // lands exactly on that canonical baseline. Its existing thresholds
        // therefore stay unchanged.
        return headPose.copy(
            pitchDeg =
                headPose.pitchDeg -
                    profile.pitchBaseDeg +
                    poseTemporalAnalyzer.baselinePitchDeg,
            yawDeg =
                headPose.yawDeg -
                    profile.yawBaseDeg +
                    poseTemporalAnalyzer.baselineYawDeg,
            rollDeg =
                headPose.rollDeg -
                    profile.rollBaseDeg +
                    poseTemporalAnalyzer.baselineRollDeg
        )
    }

    private fun applyGazeCalibration(
        cues: FatigueCueMetrics?,
        profile: DriverCalibrationProfile?
    ): FatigueCueMetrics? {
        if (
            cues == null ||
            profile == null ||
            !cues.valid ||
            !cues.gaze.horizontal.isFinite() ||
            !cues.gaze.vertical.isFinite()
        ) {
            return cues
        }

        // Keep the current gaze thresholds unchanged. We translate the
        // personal neutral gaze to the center of the existing bands.
        val canonicalHorizontalCenter =
            (
                gazeTemporalAnalyzer.horizontalLow +
                    gazeTemporalAnalyzer.horizontalHigh
                ) / 2.0
        val canonicalVerticalCenter =
            (
                gazeTemporalAnalyzer.verticalLow +
                    gazeTemporalAnalyzer.verticalHigh
                ) / 2.0

        return cues.copy(
            gaze = cues.gaze.copy(
                horizontal =
                    (
                        cues.gaze.horizontal -
                            profile.gazeCenterH +
                            canonicalHorizontalCenter
                        ).coerceIn(0.0, 1.0),
                vertical =
                    (
                        cues.gaze.vertical -
                            profile.gazeCenterV +
                            canonicalVerticalCenter
                        ).coerceIn(0.0, 1.0)
            )
        )
    }

    private fun logAppliedCalibration(
        profile: DriverCalibrationProfile?,
        rawCues: FatigueCueMetrics?,
        rawPose: HeadPose?,
        gazeHorizontal: Double,
        gazeVertical: Double,
        posePitchRelative: Double,
        poseYawRelative: Double,
        poseRollRelative: Double
    ) {
        if (
            profile == null ||
            rawCues?.valid != true ||
            rawPose == null ||
            !gazeHorizontal.isFinite() ||
            !gazeVertical.isFinite() ||
            !posePitchRelative.isFinite() ||
            !poseYawRelative.isFinite() ||
            !poseRollRelative.isFinite()
        ) {
            return
        }

        calibrationApplyLogCount += 1L
        if (
            calibrationApplyLogCount > 3L &&
            calibrationApplyLogCount % 30L != 0L
        ) {
            return
        }

        Log.i(
            "H7_14_APPLY",
            String.format(
                Locale.US,
                "source=%s rawPose=%.2f/%.2f/%.2f relPose=%.2f/%.2f/%.2f rawGaze=%.3f/%.3f centeredGaze=%.3f/%.3f",
                profile.source.name,
                rawPose.pitchDeg,
                rawPose.yawDeg,
                rawPose.rollDeg,
                posePitchRelative,
                poseYawRelative,
                poseRollRelative,
                rawCues.gaze.horizontal,
                rawCues.gaze.vertical,
                gazeHorizontal,
                gazeVertical
            )
        )
    }

    private fun percentOrZero(value: Double): Double {
        return if (value.isFinite()) {
            value * 100.0
        } else {
            0.0
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        if (cameraInputMode != CameraInputMode.INTERNAL) {
            return
        }

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        detectFace(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectFace(imageProxy: ImageProxy) {
        val now = SystemClock.uptimeMillis()

        if (now - lastAnalysisTimeMs < ANALYSIS_INTERVAL_MS) {
            imageProxy.close()
            return
        }

        lastAnalysisTimeMs = now

        if (
            !this::faceLandmarkerHelper.isInitialized ||
            faceLandmarkerHelper.isClose()
        ) {
            imageProxy.close()
            return
        }

        try {
            faceLandmarkerHelper.detectLiveStream(
                imageProxy = imageProxy,
                isFrontCamera =
                    cameraFacing == CameraSelector.LENS_FACING_FRONT
            )
        } catch (error: RuntimeException) {
            // detectLiveStream garantiza el cierre de ImageProxy.
            Log.e(TAG, "Live stream analysis failed", error)
            onError(
                error.message ?: "Live stream analysis failed"
            )
        }
    }

    private fun switchCameraInput() {
        when (cameraInputMode) {
            CameraInputMode.INTERNAL -> {
                cameraInputMode = CameraInputMode.USB
                resetTemporalAnalyzers()
                if (this::faceLandmarkerHelper.isInitialized) {
                    faceLandmarkerHelper.resetLiveStreamTracking()
                }
                fragmentCameraBinding.overlay.clear()
                fragmentCameraBinding.overlay.invalidate()
                cameraProvider?.unbindAll()
                camera = null
                imageAnalyzer = null
                preview = null

                fragmentCameraBinding.viewFinder.visibility =
                    View.GONE
                fragmentCameraBinding.usbViewFinder.visibility =
                    View.VISIBLE
                fragmentCameraBinding.cameraSourceToggle
                    .setText(R.string.camera_source_usb)

                val surface =
                    usbSurface
                        ?: fragmentCameraBinding
                            .usbViewFinder
                            .holder
                            .surface

                if (surface.isValid) {
                    startUsbCamera(surface)
                } else {
                    Log.i(
                        "H7_13_UVC",
                        "Waiting for USB SurfaceView"
                    )
                }
            }

            CameraInputMode.USB -> {
                cameraInputMode =
                    CameraInputMode.INTERNAL
                stopUsbCamera()
                resetTemporalAnalyzers()
                if (this::faceLandmarkerHelper.isInitialized) {
                    faceLandmarkerHelper.resetLiveStreamTracking()
                }
                fragmentCameraBinding.overlay.clear()
                fragmentCameraBinding.overlay.invalidate()

                fragmentCameraBinding
                    .usbViewFinder
                    .visibility = View.GONE
                fragmentCameraBinding
                    .viewFinder
                    .visibility = View.VISIBLE
                fragmentCameraBinding
                    .cameraSourceToggle
                    .setText(
                        R.string.camera_source_internal
                    )

                if (cameraProvider != null) {
                    bindCameraUseCases()
                } else {
                    setUpCamera()
                }
            }
        }
    }

    private fun startUsbCamera(surface: Surface) {
        if (
            cameraInputMode != CameraInputMode.USB ||
            !surface.isValid
        ) {
            return
        }

        fragmentCameraBinding.cameraSourceToggle
            .setText(R.string.camera_source_usb)
        Log.i(
            "H7_13_UVC",
            "Starting direct UVC source"
        )
        usbCameraSource.start(surface)
    }

    private fun stopUsbCamera() {
        if (this::usbCameraSource.isInitialized) {
            usbCameraSource.stop()
        }
        usbFrameInFlight.set(false)
        Log.i(
            "H7_13_UVC",
            "Direct UVC source stopped"
        )
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after face have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: FaceLandmarkerHelper.ResultBundle
    ) {
        performanceTelemetry.record(
            delegate = faceLandmarkerHelper.activeDelegate,
            endToEndMs = resultBundle.inferenceTime,
            preprocessingMs =
                resultBundle.preprocessingTime,
            mediaPipeMs =
                resultBundle.mediaPipeInferenceTime
        )

        val timestampMs = resultBundle.result.timestampMs()
        val cueMetrics = resultBundle.result
            .faceLandmarks()
            .firstOrNull()
            ?.let { FatigueCues.fromLandmarks(it) }

        val transformationMatrix: FloatArray? =
            resultBundle.result
                .facialTransformationMatrixes()
                .orElse(emptyList<FloatArray>())
                .firstOrNull()

        val headPose = transformationMatrix?.let { matrix ->
            runCatching {
                HeadPoseEstimator.fromColumnMajor(matrix)
            }.getOrNull()
        }

        consumeNeutralCalibrationSample(
            timestampMs = timestampMs,
            cueMetrics = cueMetrics,
            headPose = headPose
        )

        val calibrationProfile =
            activeCalibrationProfile()
        val calibratedHeadPose =
            applyPoseCalibration(
                headPose = headPose,
                profile = calibrationProfile
            )
        val calibratedCueMetrics =
            applyGazeCalibration(
                cues = cueMetrics,
                profile = calibrationProfile
            )

        val poseMetrics = synchronized(temporalLock) {
            poseTemporalAnalyzer.update(
                timestampMs = timestampMs,
                pitchDeg =
                    calibratedHeadPose?.pitchDeg
                        ?: Double.NaN,
                yawDeg =
                    calibratedHeadPose?.yawDeg
                        ?: Double.NaN,
                rollDeg =
                    calibratedHeadPose?.rollDeg
                        ?: Double.NaN,
                valid =
                    calibratedHeadPose != null &&
                        calibratedHeadPose.fitError <= 0.10
            )
        }

        val temporalMetrics = synchronized(temporalLock) {
            val geometryValid = cueMetrics?.valid == true
            val eye = eyeTemporalAnalyzer.update(
                timestampMs = timestampMs,
                ear = cueMetrics?.ear ?: Double.NaN,
                valid = geometryValid
            )
            val mouth = mouthTemporalAnalyzer.update(
                timestampMs = timestampMs,
                mar = cueMetrics?.mar ?: Double.NaN,
                valid = geometryValid
            )
            val gazeValid = cueMetrics?.let { cues ->
                geometryValid &&
                    cues.ear >= MOBILE_EAR_THRESHOLD &&
                    cues.gaze.horizontalAsymmetry <=
                    MAX_GAZE_ASYMMETRY
            } ?: false
            val gaze = gazeTemporalAnalyzer.update(
                timestampMs = timestampMs,
                horizontal =
                    calibratedCueMetrics
                        ?.gaze
                        ?.horizontal
                        ?: Double.NaN,
                vertical =
                    calibratedCueMetrics
                        ?.gaze
                        ?.vertical
                        ?: Double.NaN,
                valid = gazeValid
            )

            Triple(eye, mouth, gaze)
        }

        val eyeMetrics = temporalMetrics.first
        val mouthMetrics = temporalMetrics.second
        val gazeMetrics = temporalMetrics.third

        // Perception is safety-critical. If a new eye-closure episode
        // begins while the optional local LLM is still generating from
        // a previous alert, stop that generation so MediaPipe and the
        // temporal fatigue pipeline regain CPU immediately. The guard
        // is armed again only after the eyes reopen, so the LLM started
        // by the current prolonged-closure alert is not cancelled by
        // subsequent frames of that same closure.
        if (
            eyeMetrics.closureDurationS >= 0.50 &&
            llmClosurePriorityCancelArmed
        ) {
            if (
                this::localLlmEngine.isInitialized &&
                localLlmEngine.isGenerationInFlight()
            ) {
                localLlmEngine.cancelGeneration(
                    reason = "perception_eye_closure"
                )
                Log.i(
                    "H7_11_ASYNC",
                    "PERCEPTION_PRIORITY_CANCEL closure_s=" +
                        String.format(
                            Locale.US,
                            "%.2f",
                            eyeMetrics.closureDurationS
                        )
                )
            }
            llmClosurePriorityCancelArmed = false
        } else if (eyeMetrics.closureDurationS <= 0.10) {
            llmClosurePriorityCancelArmed = true
            flushDeferredClosureLlmIfReady(
                eyeMetrics.closureDurationS
            )
        }

        logAppliedCalibration(
            profile = calibrationProfile,
            rawCues = cueMetrics,
            rawPose = headPose,
            gazeHorizontal = gazeMetrics.horizontal,
            gazeVertical = gazeMetrics.vertical,
            posePitchRelative =
                poseMetrics.pitchRelativeDeg,
            poseYawRelative =
                poseMetrics.yawRelativeDeg,
            poseRollRelative =
                poseMetrics.rollRelativeDeg
        )

        val engineMetrics = synchronized(temporalLock) {
            val faceValid = cueMetrics?.valid == true
            val eyeValid =
                faceValid &&
                    eyeMetrics.perclos.isFinite()
            val gazeValid =
                gazeMetrics.direction != "NO_VALIDA"
            val poseValid =
                poseMetrics.direction != "NO_VALIDA"

            fatigueEngine.update(
                timestampMs = timestampMs,
                closureS =
                    eyeMetrics.closureDurationS,
                perclosRatio = eyeMetrics.perclos,
                gazeAwayS =
                    gazeMetrics.awayDurationS,
                yawnEvent =
                    mouthMetrics.yawnEvent,
                nodEvent = poseMetrics.nodEvent,
                faceValid = faceValid,
                eyeValid = eyeValid,
                mouthValid = faceValid,
                gazeValid = gazeValid,
                poseValid = poseValid,
                acknowledged =
                    acknowledgementRequested
                        .getAndSet(false)
            )
        }

        writeH8ExperimentFrame(
            H8AblationFrame(
                timestampMs = timestampMs,
                source = calibrationSource(),
                faceDetected =
                    cueMetrics?.valid == true,
                validEye =
                    cueMetrics?.valid == true &&
                        eyeMetrics.perclos.isFinite(),
                validMouth =
                    cueMetrics?.valid == true,
                validGaze =
                    gazeMetrics.direction != "NO_VALIDA",
                validPose =
                    poseMetrics.direction != "NO_VALIDA",
                ear = cueMetrics?.ear ?: Double.NaN,
                mar = cueMetrics?.mar ?: Double.NaN,
                gazeHRaw =
                    cueMetrics?.gaze?.horizontal
                        ?: Double.NaN,
                gazeVRaw =
                    cueMetrics?.gaze?.vertical
                        ?: Double.NaN,
                gazeHAsymmetry =
                    cueMetrics?.gaze?.horizontalAsymmetry
                        ?: Double.NaN,
                gazeVAsymmetry =
                    cueMetrics?.gaze?.verticalAsymmetry
                        ?: Double.NaN,
                pitchDeg =
                    headPose?.pitchDeg ?: Double.NaN,
                yawDeg =
                    headPose?.yawDeg ?: Double.NaN,
                rollDeg =
                    headPose?.rollDeg ?: Double.NaN,
                calibrationProfile =
                    calibrationProfile,
                ablationMode =
                    engineMetrics.ablationMode,
                score = engineMetrics.score,
                state = engineMetrics.state,
                stateEvent = engineMetrics.stateEvent,
                closureS =
                    eyeMetrics.closureDurationS,
                perclos = eyeMetrics.perclos,
                yawnEvent =
                    mouthMetrics.yawnEvent,
                gazeAwayS =
                    gazeMetrics.awayDurationS,
                nodEvent = poseMetrics.nodEvent
            )
        )

        val historyTimestampMs =
            System.currentTimeMillis()
        val historyRecords = eventRecorder.record(
            FatigueHistoryFrame(
                timestampMs = historyTimestampMs,
                faceValid = cueMetrics?.valid == true,
                state = engineMetrics.state,
                stateEvent = engineMetrics.stateEvent,
                score = engineMetrics.score,
                reasons = engineMetrics.reasons,
                closureS =
                    eyeMetrics.closureDurationS,
                perclosRatio = eyeMetrics.perclos,
                blinkCount = eyeMetrics.blinkCount,
                yawnEvent =
                    mouthMetrics.yawnEvent,
                yawnsLast5Min =
                    engineMetrics.yawnsLast5Min,
                nodEvent = poseMetrics.nodEvent,
                nodsLast2Min =
                    engineMetrics.nodsLast2Min,
                gazeAwayEvent =
                    gazeMetrics.gazeAwayEvent,
                gazeAwayCount =
                    gazeMetrics.gazeAwayCount,
                poseAwayEvent =
                    poseMetrics.poseAwayEvent,
                poseAwayCount =
                    poseMetrics.poseAwayCount
            )
        )
        logHistoryRecords(historyRecords)
        evaluateContextualAlert(
            FatigueAlertSnapshot(
                timestampMs = historyTimestampMs,
                state = engineMetrics.state,
                stateEvent = engineMetrics.stateEvent,
                score = engineMetrics.score,
                reasons = engineMetrics.reasons,
                closureS =
                    eyeMetrics.closureDurationS,
                perclosRatio =
                    eyeMetrics.perclos.takeIf {
                        it.isFinite()
                    } ?: 0.0,
                yawnsLast5Min =
                    engineMetrics.yawnsLast5Min,
                nodsLast2Min =
                    engineMetrics.nodsLast2Min,
                faceValid = cueMetrics?.valid == true,
                dataSufficient =
                    engineMetrics.dataSufficient,
                alertAcknowledged =
                    engineMetrics.alertAcknowledged
            )
        )

        if (engineMetrics.stateEvent != null) {
            Log.i(
                H8_2_ABLATION_TAG,
                "EVENT mode=" +
                    engineMetrics.ablationMode.code +
                    " state=" +
                    engineMetrics.state.visibleLabel +
                    " score=" +
                    String.format(
                        Locale.US,
                        "%.3f",
                        engineMetrics.score
                    ) +
                    " event=" +
                    engineMetrics.stateEvent
            )
        }

        if (engineMetrics.stateEvent == "state_normal") {
            synchronized(temporalLock) {
                eyeTemporalAnalyzer.reset()
                mouthTemporalAnalyzer.reset()
                gazeTemporalAnalyzer.reset()
                poseTemporalAnalyzer.reset()
            }
            Log.i(
                "FatigueEngine",
                "ACTIVE_EVIDENCE_RESET_AFTER_NORMAL"
            )
        }

        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null && isAdded) {
                fragmentCameraBinding.fatigueMetrics.text =
                    if (cueMetrics?.valid == true) {
                        getString(
                            R.string.fatigue_metrics_values,
                            cueMetrics.ear,
                            eyeMetrics.closureDurationS,
                            percentOrZero(eyeMetrics.perclos),
                            eyeMetrics.blinkCount,
                            cueMetrics.mar,
                            mouthMetrics.openingDurationS,
                            mouthMetrics.yawnCount,
                            gazeMetrics.horizontal,
                            gazeMetrics.vertical,
                            gazeMetrics.direction,
                            gazeMetrics.awayDurationS,
                            gazeMetrics.gazeAwayCount,
                            percentOrZero(
                                eyeMetrics.validRatio
                            ),
                            headPose?.pitchDeg
                                ?: Double.NaN,
                            headPose?.yawDeg
                                ?: Double.NaN,
                            headPose?.rollDeg
                                ?: Double.NaN,
                            headPose?.scale
                                ?: Double.NaN,
                            headPose?.fitError
                                ?: Double.NaN,
                            poseMetrics.pitchRelativeDeg,
                            poseMetrics.yawRelativeDeg,
                            poseMetrics.rollRelativeDeg,
                            poseMetrics.direction,
                            poseMetrics.awayDurationS,
                            poseMetrics.poseAwayCount,
                            poseMetrics.nodDurationS,
                            poseMetrics.nodCount,
                            engineMetrics.score,
                            engineMetrics.state.visibleLabel,
                            if (
                                engineMetrics.reasons.isEmpty()
                            ) {
                                "ninguna"
                            } else {
                                engineMetrics.reasons
                                    .joinToString(",")
                            },
                            engineMetrics.yawnsLast5Min,
                            engineMetrics.nodsLast2Min,
                            engineMetrics
                                .availableWeightRatio *
                                100.0,
                            when {
                                engineMetrics.state ==
                                    FatigueState.CRITICAL &&
                                    !engineMetrics
                                        .alertAcknowledged ->
                                    getString(
                                        R.string
                                            .fatigue_ack_hint
                                    )
                                engineMetrics.state ==
                                    FatigueState.CRITICAL ->
                                    String.format(
                                        Locale.US,
                                        "Recuperación: %.1f / 30 s",
                                        engineMetrics
                                            .criticalRecoveryElapsedS
                                    )
                                else -> ""
                            }
                        )
                    } else {
                        getString(
                            R.string.fatigue_metrics_invalid
                        )
                    }

                if (
                    fragmentCameraBinding
                        .recyclerviewResults
                        .scrollState != SCROLL_STATE_DRAGGING
                ) {
                    faceBlendshapesResultAdapter
                        .updateResults(resultBundle.result)
                    faceBlendshapesResultAdapter
                        .notifyDataSetChanged()
                }

                fragmentCameraBinding
                    .bottomSheetLayout
                    .inferenceTimeVal
                    .text = String.format(
                    Locale.US,
                    "%d ms",
                    resultBundle.inferenceTime
                )

                fragmentCameraBinding.overlay.setResults(
                    resultBundle.result,
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )
                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

    override fun onEmpty(
        inferenceTime: Long,
        timestampMs: Long,
        preprocessingTime: Long,
        mediaPipeInferenceTime: Long
    ) {
        performanceTelemetry.record(
            delegate = faceLandmarkerHelper.activeDelegate,
            endToEndMs = inferenceTime,
            preprocessingMs = preprocessingTime,
            mediaPipeMs = mediaPipeInferenceTime
        )
        val poseMetrics = synchronized(temporalLock) {
            poseTemporalAnalyzer.update(
                timestampMs,
                Double.NaN,
                Double.NaN,
                Double.NaN,
                false
            )
        }

        val temporalMetrics = synchronized(temporalLock) {
            val eye = eyeTemporalAnalyzer.update(
                timestampMs,
                Double.NaN,
                false
            )
            val mouth = mouthTemporalAnalyzer.update(
                timestampMs,
                Double.NaN,
                false
            )
            val gaze = gazeTemporalAnalyzer.update(
                timestampMs,
                Double.NaN,
                Double.NaN,
                false
            )
            Triple(eye, mouth, gaze)
        }

        val engineMetrics = synchronized(temporalLock) {
            fatigueEngine.update(
                timestampMs = timestampMs,
                closureS = Double.NaN,
                perclosRatio = Double.NaN,
                gazeAwayS = Double.NaN,
                yawnEvent = false,
                nodEvent = false,
                faceValid = false,
                eyeValid = false,
                mouthValid = false,
                gazeValid = false,
                poseValid = false
            )
        }

        // H8 schema v2: log no-face/onEmpty frames as part of the
        // experimental timeline. These frames invalidate pending state
        // transitions on Android and must therefore exist in the offline
        // A-E replay as well.
        writeH8ExperimentFrame(
            H8AblationFrame(
                timestampMs = timestampMs,
                source = calibrationSource(),
                faceDetected = false,
                validEye = false,
                validMouth = false,
                validGaze = false,
                validPose = false,
                ear = Double.NaN,
                mar = Double.NaN,
                gazeHRaw = Double.NaN,
                gazeVRaw = Double.NaN,
                gazeHAsymmetry = Double.NaN,
                gazeVAsymmetry = Double.NaN,
                pitchDeg = Double.NaN,
                yawDeg = Double.NaN,
                rollDeg = Double.NaN,
                calibrationProfile =
                    activeCalibrationProfile(),
                ablationMode =
                    engineMetrics.ablationMode,
                score = engineMetrics.score,
                state = engineMetrics.state,
                stateEvent = engineMetrics.stateEvent,
                closureS =
                    temporalMetrics.first.closureDurationS,
                perclos =
                    temporalMetrics.first.perclos,
                yawnEvent = false,
                gazeAwayS =
                    temporalMetrics.third.awayDurationS,
                nodEvent = false
            )
        )

        val historyTimestampMs =
            System.currentTimeMillis()
        val historyRecords = eventRecorder.record(
            FatigueHistoryFrame(
                timestampMs = historyTimestampMs,
                faceValid = false,
                state = engineMetrics.state,
                stateEvent = engineMetrics.stateEvent,
                score = engineMetrics.score,
                reasons = engineMetrics.reasons,
                closureS = 0.0,
                perclosRatio =
                    temporalMetrics.first.perclos,
                blinkCount =
                    temporalMetrics.first.blinkCount,
                yawnEvent = false,
                yawnsLast5Min =
                    engineMetrics.yawnsLast5Min,
                nodEvent = false,
                nodsLast2Min =
                    engineMetrics.nodsLast2Min,
                gazeAwayEvent = false,
                gazeAwayCount =
                    temporalMetrics.third.gazeAwayCount,
                poseAwayEvent = false,
                poseAwayCount =
                    poseMetrics.poseAwayCount
            )
        )
        logHistoryRecords(historyRecords)
        evaluateContextualAlert(
            FatigueAlertSnapshot(
                timestampMs = historyTimestampMs,
                state = engineMetrics.state,
                stateEvent = engineMetrics.stateEvent,
                score = engineMetrics.score,
                reasons = engineMetrics.reasons,
                closureS = 0.0,
                perclosRatio = 0.0,
                yawnsLast5Min =
                    engineMetrics.yawnsLast5Min,
                nodsLast2Min =
                    engineMetrics.nodsLast2Min,
                faceValid = false,
                dataSufficient = false,
                alertAcknowledged =
                    engineMetrics.alertAcknowledged
            )
        )

        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null && isAdded) {
                fragmentCameraBinding.overlay.clear()
                fragmentCameraBinding.fatigueMetrics.text =
                    getString(
                        R.string.fatigue_metrics_no_face_values,
                        percentOrZero(
                            temporalMetrics.first.perclos
                        ),
                        temporalMetrics.first.blinkCount,
                        temporalMetrics.second.yawnCount,
                        temporalMetrics.third.gazeAwayCount,
                        percentOrZero(
                            temporalMetrics.first.validRatio
                        ),
                        poseMetrics.poseAwayCount,
                        poseMetrics.nodCount,
                        engineMetrics.state.visibleLabel
                    )
                fragmentCameraBinding
                    .bottomSheetLayout
                    .inferenceTimeVal
                    .text = String.format(
                    Locale.US,
                    "%d ms",
                    inferenceTime
                )
                faceBlendshapesResultAdapter
                    .updateResults(null)
                faceBlendshapesResultAdapter
                    .notifyDataSetChanged()
            }
        }
    }

    private fun showDriverContextIntroIfNeeded() {
        if (driverContextPromptShownThisSession) {
            return
        }

        markDriverContextPromptShown()

        AlertDialog.Builder(requireContext())
            .setTitle(R.string.driver_context_intro_title)
            .setMessage(R.string.driver_context_intro_message)
            .setNegativeButton(
                R.string.driver_context_skip
            ) { _, _ ->
                Log.i(
                    "H8_4_CONTEXT",
                    "PRE_DRIVE_SKIPPED"
                )
            }
            .setPositiveButton(
                R.string.driver_context_add
            ) { _, _ ->
                showDriverContextDialog()
            }
            .show()
    }

    private fun markDriverContextPromptShown() {
        driverContextPromptShownThisSession = true
        requireActivity().intent.putExtra(
            EXTRA_CONTEXT_PROMPT_SHOWN,
            true
        )
    }

    private fun showDriverContextDialog() {
        val context = requireContext()
        val density = resources.displayMetrics.density
        val padding = (16f * density).toInt()
        val latest = driverContextStore.latest()

        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(padding, padding, padding, padding)
        }

        fun addLabel(textRes: Int) {
            container.addView(
                TextView(context).apply {
                    text = getString(textRes)
                    setPadding(0, padding, 0, 0)
                }
            )
        }

        fun decimalInput(
            labelRes: Int,
            hintRes: Int,
            value: Double?
        ): EditText {
            addLabel(labelRes)
            return EditText(context).apply {
                hint = getString(hintRes)
                inputType =
                    InputType.TYPE_CLASS_NUMBER or
                        InputType.TYPE_NUMBER_FLAG_DECIMAL
                value?.let {
                    setText(
                        String.format(
                            Locale.US,
                            "%.1f",
                            it
                        )
                    )
                }
                container.addView(this)
            }
        }

        val sleepLastInput = decimalInput(
            R.string.driver_context_sleep_last_label,
            R.string.driver_context_sleep_last_hint,
            latest?.sleepLastNightHours
        )

        val sleepAverageInput = decimalInput(
            R.string.driver_context_sleep_avg_label,
            R.string.driver_context_sleep_avg_hint,
            latest?.sleepAverage3dHours
        )

        addLabel(R.string.driver_context_wake_label)
        val wakeInput = EditText(context).apply {
            hint = getString(
                R.string.driver_context_wake_hint
            )
            inputType =
                InputType.TYPE_CLASS_DATETIME or
                    InputType.TYPE_DATETIME_VARIATION_TIME
            latest?.wakeTimestampMs?.let {
                val calendar = Calendar.getInstance().apply {
                    timeInMillis = it
                }
                setText(
                    String.format(
                        Locale.US,
                        "%02d:%02d",
                        calendar.get(Calendar.HOUR_OF_DAY),
                        calendar.get(Calendar.MINUTE)
                    )
                )
            }
        }
        container.addView(wakeInput)

        addLabel(R.string.driver_context_sleepiness_label)
        val sleepinessOptions =
            listOf(
                getString(
                    R.string.driver_context_not_reported
                )
            ) +
                (1..9).map { it.toString() }
        val sleepinessSpinner = Spinner(context).apply {
            adapter = ArrayAdapter(
                context,
                android.R.layout
                    .simple_spinner_dropdown_item,
                sleepinessOptions
            )
            setSelection(
                latest?.subjectiveSleepiness ?: 0
            )
        }
        container.addView(sleepinessSpinner)

        val shiftInput = decimalInput(
            R.string.driver_context_shift_label,
            R.string.driver_context_shift_hint,
            latest?.workShiftHours
        )

        addLabel(R.string.driver_context_night_shift_label)
        val yesNoOptions = listOf(
            getString(R.string.driver_context_not_reported),
            getString(R.string.driver_context_no),
            getString(R.string.driver_context_yes)
        )
        val nightShiftSpinner = Spinner(context).apply {
            adapter = ArrayAdapter(
                context,
                android.R.layout
                    .simple_spinner_dropdown_item,
                yesNoOptions
            )
            setSelection(
                when (latest?.nightShift) {
                    false -> 1
                    true -> 2
                    null -> 0
                }
            )
        }
        container.addView(nightShiftSpinner)

        val demandOptions =
            listOf(
                getString(
                    R.string.driver_context_not_reported
                )
            ) +
                RecentActivityLevel.values().map {
                    it.displayLabel
                }

        addLabel(R.string.driver_context_physical_label)
        val physicalSpinner = Spinner(context).apply {
            adapter = ArrayAdapter(
                context,
                android.R.layout
                    .simple_spinner_dropdown_item,
                demandOptions
            )
            setSelection(
                latest?.physicalDemand?.let {
                    RecentActivityLevel.values()
                        .indexOf(it) + 1
                } ?: 0
            )
        }
        container.addView(physicalSpinner)

        addLabel(R.string.driver_context_mental_label)
        val mentalSpinner = Spinner(context).apply {
            adapter = ArrayAdapter(
                context,
                android.R.layout
                    .simple_spinner_dropdown_item,
                demandOptions
            )
            setSelection(
                latest?.mentalDemand?.let {
                    RecentActivityLevel.values()
                        .indexOf(it) + 1
                } ?: 0
            )
        }
        container.addView(mentalSpinner)

        val scrollView = ScrollView(context).apply {
            addView(container)
        }

        val dialog = AlertDialog.Builder(context)
            .setTitle(R.string.driver_context_title)
            .setView(scrollView)
            .setNegativeButton(
                R.string.driver_context_cancel,
                null
            )
            .setPositiveButton(
                R.string.driver_context_save,
                null
            )
            .create()

        dialog.setOnShowListener {
            dialog.getButton(
                AlertDialog.BUTTON_POSITIVE
            ).setOnClickListener {
                val nowMs = System.currentTimeMillis()

                val sleepLast =
                    parseOptionalHours(
                        sleepLastInput.text.toString()
                    )
                val sleepAverage =
                    parseOptionalHours(
                        sleepAverageInput.text.toString()
                    )
                val wakeTimestamp =
                    parseWakeTimestamp(
                        wakeInput.text.toString(),
                        nowMs
                    )
                val sleepiness =
                    sleepinessSpinner
                        .selectedItemPosition
                        .takeIf { it in 1..9 }
                val shift =
                    parseOptionalHours(
                        shiftInput.text.toString()
                    )
                val nightShift =
                    when (
                        nightShiftSpinner
                            .selectedItemPosition
                    ) {
                        1 -> false
                        2 -> true
                        else -> null
                    }
                val physicalDemand =
                    demandFromSelection(
                        physicalSpinner
                            .selectedItemPosition
                    )
                val mentalDemand =
                    demandFromSelection(
                        mentalSpinner
                            .selectedItemPosition
                    )

                val sleepLastRaw =
                    sleepLastInput.text.toString()
                val sleepAverageRaw =
                    sleepAverageInput.text.toString()
                val shiftRaw =
                    shiftInput.text.toString()
                val wakeRaw =
                    wakeInput.text.toString()

                sleepLastInput.error = null
                sleepAverageInput.error = null
                shiftInput.error = null
                wakeInput.error = null

                var invalid = false

                if (
                    invalidHours(
                        sleepLastRaw,
                        sleepLast
                    )
                ) {
                    sleepLastInput.error =
                        getString(
                            R.string
                                .driver_context_invalid_hours
                        )
                    invalid = true
                }

                if (
                    invalidHours(
                        sleepAverageRaw,
                        sleepAverage
                    )
                ) {
                    sleepAverageInput.error =
                        getString(
                            R.string
                                .driver_context_invalid_hours
                        )
                    invalid = true
                }

                if (
                    invalidHours(
                        shiftRaw,
                        shift
                    )
                ) {
                    shiftInput.error =
                        getString(
                            R.string
                                .driver_context_invalid_hours
                        )
                    invalid = true
                }

                if (
                    wakeRaw.trim().isNotEmpty() &&
                    wakeTimestamp == null
                ) {
                    wakeInput.error =
                        getString(
                            R.string
                                .driver_context_invalid_wake
                        )
                    invalid = true
                }

                val empty =
                    sleepLast == null &&
                        sleepAverage == null &&
                        wakeTimestamp == null &&
                        sleepiness == null &&
                        shift == null &&
                        nightShift == null &&
                        physicalDemand == null &&
                        mentalDemand == null

                if (invalid) {
                    Log.w(
                        "H8_4_CONTEXT",
                        "VALIDATION_FAILED " +
                            "sleep_last_raw='" +
                            sleepLastRaw + "' " +
                            "sleep_avg_raw='" +
                            sleepAverageRaw + "' " +
                            "wake_raw='" +
                            wakeRaw + "' " +
                            "shift_raw='" +
                            shiftRaw + "'"
                    )
                    Toast.makeText(
                        context,
                        R.string.driver_context_fix_marked,
                        Toast.LENGTH_SHORT
                    ).show()
                    return@setOnClickListener
                }

                if (empty) {
                    Toast.makeText(
                        context,
                        R.string.driver_context_invalid,
                        Toast.LENGTH_SHORT
                    ).show()
                    return@setOnClickListener
                }

                driverContextStore.append(
                    DriverContextEntry(
                        timestampMs = nowMs,
                        sleepLastNightHours =
                            sleepLast,
                        sleepAverage3dHours =
                            sleepAverage,
                        wakeTimestampMs =
                            wakeTimestamp,
                        subjectiveSleepiness =
                            sleepiness,
                        workShiftHours = shift,
                        nightShift = nightShift,
                        physicalDemand =
                            physicalDemand,
                        mentalDemand = mentalDemand
                    )
                )

                Log.i(
                    "H8_4_CONTEXT",
                    "SAVED sleep_last_night_h=" +
                        (sleepLast ?: "NA") +
                        " sleep_avg_3d_h=" +
                        (sleepAverage ?: "NA") +
                        " wake_ts=" +
                        (wakeTimestamp ?: "NA") +
                        " sleepiness_1_9=" +
                        (sleepiness ?: "NA") +
                        " work_shift_h=" +
                        (shift ?: "NA") +
                        " night_shift=" +
                        (nightShift ?: "NA") +
                        " physical_demand=" +
                        (
                            physicalDemand
                                ?.wireValue ?: "NA"
                            ) +
                        " mental_demand=" +
                        (
                            mentalDemand
                                ?.wireValue ?: "NA"
                            )
                )

                Toast.makeText(
                    context,
                    R.string.driver_context_saved,
                    Toast.LENGTH_SHORT
                ).show()
                dialog.dismiss()
            }
        }

        dialog.show()
    }

    private fun parseOptionalHours(
        rawValue: String
    ): Double? {
        val normalized = rawValue
            .replace('\u00A0', ' ')
            .trim()
            .replace(" ", "")
            .replace(',', '.')

        if (normalized.isEmpty()) {
            return null
        }

        return normalized.toDoubleOrNull()
    }

    private fun invalidHours(
        rawValue: String,
        parsedValue: Double?
    ): Boolean {
        val trimmed = rawValue.trim()
        if (trimmed.isEmpty()) {
            return false
        }

        return parsedValue == null ||
            !parsedValue.isFinite() ||
            parsedValue !in 0.0..24.0
    }

    private fun parseWakeTimestamp(
        rawValue: String,
        nowMs: Long
    ): Long? {
        val trimmed = rawValue.trim()

        if (trimmed.isEmpty()) {
            return null
        }

        val parts = trimmed.split(":")
        if (parts.size != 2) {
            return null
        }

        val hour = parts[0].toIntOrNull()
            ?: return null
        val minute = parts[1].toIntOrNull()
            ?: return null

        if (hour !in 0..23 || minute !in 0..59) {
            return null
        }

        val wake = Calendar.getInstance().apply {
            timeInMillis = nowMs
            set(Calendar.HOUR_OF_DAY, hour)
            set(Calendar.MINUTE, minute)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }

        if (wake.timeInMillis > nowMs) {
            wake.add(Calendar.DAY_OF_YEAR, -1)
        }

        return wake.timeInMillis
    }

    private fun demandFromSelection(
        position: Int
    ): RecentActivityLevel? {
        if (position <= 0) {
            return null
        }

        return RecentActivityLevel.values()
            .getOrNull(position - 1)
    }

    private fun evaluateContextualAlert(
        snapshot: FatigueAlertSnapshot
    ) {
        val baseDecision =
            contextualAlertPlanner.evaluate(
                snapshot = snapshot,
                historyRecords =
                    historyStore.recent(500)
            ) ?: return

        val sessionDurationMinutes =
            (
                SystemClock.elapsedRealtime() -
                    contextualSessionStartedElapsedMs
                )
                .coerceAtLeast(0L) /
                60_000L
        val userContext =
            driverContextStore.summarize(
                nowMs = snapshot.timestampMs
            )
        val userContextFacts =
            userContext.toPromptFacts(
                sessionDurationMinutes =
                    sessionDurationMinutes,
                nowMs = snapshot.timestampMs
            )
        val decision = baseDecision.copy(
            languageContext =
                baseDecision.languageContext.copy(
                    historicalFacts =
                        baseDecision.languageContext
                            .historicalFacts +
                            userContextFacts
                )
        )

        Log.i(
            "H8_4_CONTEXT",
            "APPLY " +
                userContextFacts.joinToString(" ")
        )

        Log.i(
            ALERT_CONTEXT_TAG,
            "Emitir " + decision.urgency +
                " | intención=" +
                decision.intent +
                " | disparador=" +
                decision.trigger +
                " | mensaje=" +
                decision.languageContext
                    .fallbackMessage
        )

        val toneDelayMs =
            when (decision.trigger) {
                "state_fatigue" -> {
                    if (this::alertSoundPlayer.isInitialized) {
                        alertSoundPlayer.playModerate()
                    }
                    FatigueAlertSoundPlayer.MODERATE_DELAY_MS
                }

                "state_critical",
                "prolonged_eye_closure" -> {
                    if (this::alertSoundPlayer.isInitialized) {
                        alertSoundPlayer.playCritical()
                    }
                    FatigueAlertSoundPlayer.CRITICAL_DELAY_MS
                }

                else -> 0L
            }

        val speakDeterministic = {
            val accepted =
                if (this::alertSpeaker.isInitialized) {
                    alertSpeaker.speak(
                        text = decision.languageContext
                            .fallbackMessage,
                        interrupt =
                            decision.urgency ==
                                AlertUrgency.CRITICAL ||
                                decision.trigger ==
                                    "face_visible_again"
                    )
                } else {
                    false
                }

            Log.i(
                "H7_11_ASYNC",
                "DETERMINISTIC_TTS accepted=" +
                    accepted +
                    " trigger=" + decision.trigger +
                    " urgency=" + decision.urgency
            )
        }

        if (toneDelayMs > 0L) {
            activity?.runOnUiThread {
                fragmentCameraBinding.root.postDelayed(
                    { speakDeterministic() },
                    toneDelayMs
                )
            }
        } else {
            speakDeterministic()
        }

        if (decision.trigger == "face_visible_again") {
            Log.i(
                "H7_11_ASYNC",
                "LLM_SKIPPED_TRANSIENT trigger=" +
                    decision.trigger
            )
            return
        }

        val llmEligible =
            decision.urgency == AlertUrgency.HIGH ||
                decision.urgency == AlertUrgency.CRITICAL

        if (!llmEligible) {
            Log.i(
                "H7_11_ASYNC",
                "LLM_SKIPPED_POLICY trigger=" +
                    decision.trigger +
                    " urgency=" +
                    decision.urgency
            )
            return
        }

        if (decision.trigger == "prolonged_eye_closure") {
            deferredClosureLlmDecision = decision
            Log.i(
                "H7_11_ASYNC",
                "LLM_DEFERRED trigger=" +
                    decision.trigger +
                    " until=eyes_reopened"
            )
            return
        }

        val llmSubmitted =
            submitContextualLlm(decision)

        Log.i(
            "H7_11_ASYNC",
            "LLM_SUBMITTED=$llmSubmitted " +
                "trigger=" + decision.trigger
        )
    }

    private fun flushDeferredClosureLlmIfReady(
        closureS: Double
    ) {
        val decision =
            deferredClosureLlmDecision ?: return

        if (closureS > 0.10) {
            return
        }

        val submitted =
            submitContextualLlm(decision)

        if (submitted) {
            deferredClosureLlmDecision = null
        }

        Log.i(
            "H7_11_ASYNC",
            "LLM_DEFERRED_SUBMIT=$submitted " +
                "trigger=" + decision.trigger +
                " closure_s=" +
                String.format(
                    Locale.US,
                    "%.2f",
                    closureS
                )
        )
    }

    private fun submitContextualLlm(
        decision: ContextualAlertDecision
    ): Boolean {
        if (!this::localLlmEngine.isInitialized) {
            return false
        }

        return localLlmEngine.submit(decision) {
            safeMessage ->
            if (!safeMessage.usedLocalModel) {
                return@submit
            }

            activity?.runOnUiThread {
                if (
                    isAdded &&
                    this::alertSpeaker.isInitialized
                ) {
                    val accepted =
                        alertSpeaker.speak(
                            text = safeMessage.text,
                            interrupt = false
                        )
                    Log.i(
                        "H7_11_ASYNC",
                        "CONTEXTUAL_TTS accepted=" +
                            accepted +
                            " status=" +
                            safeMessage.status +
                            " trigger=" +
                            decision.trigger
                    )
                }
            }
        }
    }

    private fun logHistoryRecords(
        records: List<com.google.mediapipe.examples.facelandmarker.FatigueHistoryRecord>
    ) {
        records.forEach { record ->
            Log.i(
                HISTORY_TAG,
                "Guardado " + record.eventType +
                    " | estado=" +
                    record.state.visibleLabel +
                    " | score=" +
                    String.format(
                        Locale.US,
                        "%.3f",
                        record.score
                    ) +
                    " | total=" +
                    historyStore.count()
            )
        }
    }

    override fun onError(error: String, errorCode: Int) {
        resetTemporalAnalyzers()
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null && isAdded) {
                fragmentCameraBinding.fatigueMetrics.text =
                    getString(R.string.fatigue_metrics_error)
                Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
                faceBlendshapesResultAdapter.updateResults(null)
                faceBlendshapesResultAdapter.notifyDataSetChanged()

                if (errorCode == FaceLandmarkerHelper.GPU_ERROR) {
                    fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                        FaceLandmarkerHelper.DELEGATE_CPU, false
                    )
                }
            }
        }
    }
}
