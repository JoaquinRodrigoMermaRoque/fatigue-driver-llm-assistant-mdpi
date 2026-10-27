package com.google.mediapipe.examples.facelandmarker

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OfflineVoiceSelectorTest {

    @Test
    fun prefersPeruvianOfflineVoice() {
        val selected = OfflineVoiceSelector.select(
            listOf(
                voice("es-es", "es", "ES", 500),
                voice("es-mx", "es", "MX", 400),
                voice("es-pe", "es", "PE", 300)
            )
        )

        requireNotNull(selected)
        assertEquals("es-pe", selected.name)
    }

    @Test
    fun rejectsNetworkVoices() {
        val selected = OfflineVoiceSelector.select(
            listOf(
                voice(
                    "es-pe-network",
                    "es",
                    "PE",
                    500,
                    networkRequired = true
                ),
                voice(
                    "es-mx-offline",
                    "es",
                    "MX",
                    300
                )
            )
        )

        requireNotNull(selected)
        assertEquals(
            "es-mx-offline",
            selected.name
        )
        assertTrue(!selected.networkRequired)
    }

    @Test
    fun fallsBackToAnyOfflineSpanishVoice() {
        val selected = OfflineVoiceSelector.select(
            listOf(
                voice("en-us", "en", "US", 500),
                voice("es-ar", "es", "AR", 300)
            )
        )

        requireNotNull(selected)
        assertEquals("es-ar", selected.name)
    }

    @Test
    fun prefersQualityWithinSameCountry() {
        val selected = OfflineVoiceSelector.select(
            listOf(
                voice("es-mx-low", "es", "MX", 200),
                voice("es-mx-high", "es", "MX", 500)
            )
        )

        requireNotNull(selected)
        assertEquals(
            "es-mx-high",
            selected.name
        )
    }

    @Test
    fun returnsNullWithoutOfflineSpanish() {
        val selected = OfflineVoiceSelector.select(
            listOf(
                voice("en-us", "en", "US", 500),
                voice(
                    "es-pe-network",
                    "es",
                    "PE",
                    500,
                    networkRequired = true
                )
            )
        )

        assertNull(selected)
    }

    private fun voice(
        name: String,
        language: String,
        country: String,
        quality: Int,
        networkRequired: Boolean = false
    ): OfflineVoiceOption {
        return OfflineVoiceOption(
            name = name,
            language = language,
            country = country,
            quality = quality,
            latency = 200,
            networkRequired = networkRequired
        )
    }
}
