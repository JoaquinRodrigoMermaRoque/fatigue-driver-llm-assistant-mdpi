# H7.9 — Viabilidad y arquitectura del LLM local en Android

## Alcance actual

H7.9 evalúa si un LLM pequeño puede añadir comentarios de voz
contextualizados, completamente offline, sin sustituir la lógica de seguridad
determinista.

El motor determinista sigue siendo el único componente autorizado para decidir:

- el nivel de fatiga;
- la prioridad de la alerta;
- el motivo principal;
- la acción obligatoria;
- los tiempos de repetición y recuperación.

La alerta segura se emite inmediatamente sin esperar al LLM. El modelo se
reserva para una segunda capa contextual que puede utilizar historial,
memoria de mensajes, contexto personal autorizado y evidencia científica
local. Toda salida generada debe pasar un validador estricto y además seguir
siendo vigente para el estado actual.

El diseño completo y el protocolo de evaluación se documentan en:

```text
scripts/H7_9_CONTEXTUAL_ARCHITECTURE_AND_PROTOCOL.md
```

No se implementará JNI hasta cerrar la evaluación de rendimiento y
concurrencia.

## Componentes fijados para la evaluación

| Componente | Versión o archivo | Motivo |
|---|---|---|
| Arquitectura | ARM64 CPU | Compatible con el vivo V2058 |
| Android | API 31 | Dispositivo físico de evaluación |
| NDK | 27.3.13750724 (r27d) | Reservado para integración JNI posterior |
| CMake | 3.22.1 | Construcción nativa reproducible posterior |
| Ninja | 1.10.2 | Backend de compilación posterior |
| llama.cpp | b11073, Android ARM64 oficial | Medición sin modificar la APK |
| Modelos candidatos | Qwen3-0.6B Q4_0 y Qwen3-1.7B Q4_K_M | Comparación eficiencia/calidad |
| Contexto | 1024 tokens | Reduce memoria y latencia |
| Salida | máximo 48 tokens | Comentario breve |
| Hilos | 4 y 6 | Comparación de latencia y temperatura |
| GPU | desactivada | Línea base reproducible en CPU |
| Semilla | 42 | Comparación controlada |

Los binarios, modelos y resultados permanecen bajo `.local/` y no se
incorporan a Git.

## Resultados ya validados

### Qwen3-0.6B Q4_0

En el vivo V2058 se observó aproximadamente:

- 7.77 s de tiempo completo;
- 850 MB de pico;
- calidad lingüística insuficiente.

Se detectaron respuestas con problemas de redacción y tono, por lo que no se
acepta como generador autónomo.

### Qwen3-1.7B Q4_K_M

La primera prueba documentada registró:

- 27.16 s de tiempo completo en frío;
- 16.91 s de inferencia interna;
- 6.09 tokens/s;
- aproximadamente 2.27 GiB de pico RSS;
- mejor gramática que 0.6B.

La respuesta repitió la acción obligatoria y fue rechazada. Esto confirma la
necesidad del validador, pero una única inferencia no permite decidir todavía
si 1.7B es viable.

## H7.9.5 — Primer benchmark extendido del 1.7B

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T182000Z
MODE=process-cold
THREADS=6
TEMPERATURE=0.3
REPEATS=2
CONCURRENT_APP=0
```

Resultados:

| Repetición | Tiempo host | Prompt eval | Eval | Generación | Pico RSS | MemAvailable mínimo | SwapFree mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.520 s | 10.116 s | 3.490 s | 6.02 tok/s | 2,363,596 KiB | 3,518,164 KiB | 5,221,772 KiB | 24.9 / 24.9 °C |
| 2 | 22.770 s | 10.561 s | 3.537 s | 5.94 tok/s | 2,274,800 KiB | 3,453,836 KiB | 5,219,980 KiB | 24.9 / 25.7 °C |

Resumen de estas dos ejecuciones:

- tiempo host medio: 23.145 s;
- generación media: aproximadamente 5.98 tokens/s;
- tiempo interno total: 13.71–14.21 s;
- pico RSS observado: aproximadamente 2.17–2.25 GiB;
- memoria disponible mínima observada: aproximadamente 3.29–3.36 GiB;
- no hubo error de ejecución;
- no hubo presión apreciable sobre swap;
- no se observó calentamiento relevante en esta prueba corta.

Interpretación provisional: el 1.7B muestra rendimiento process-cold consistente
entre ambas repeticiones. La diferencia entre el tiempo interno de llama.cpp y
el tiempo observado desde el host confirma que la carga/inicialización del
proceso sigue siendo una fracción importante del coste total. Estos datos no
permiten todavía inferir el rendimiento warm ni la viabilidad conjunta con la
APK, porque cámara y MediaPipe estaban desactivados.

La siguiente prueba obligatoria es mantener el modelo residente mediante
`llama-server` y medir solicitudes warm consecutivas.

## H7.9.5 — Warm real del 1.7B

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T182635Z
MODE=warm-server
THREADS=6
TEMPERATURE=0.3
REPEATS=3
CONCURRENT_APP=0
```

Resultados:

| Solicitud | Latencia host | Pico RSS | MemAvailable mínimo | SwapFree mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|---:|
| 1 | 14.890 s | 2,276,104 KiB | 3,418,428 KiB | 5,220,492 KiB | 26.6 / 26.6 °C |
| 2 | 4.220 s | 2,275,920 KiB | 3,453,072 KiB | 5,220,492 KiB | 26.6 / 26.6 °C |
| 3 | 3.860 s | 2,275,920 KiB | 3,491,920 KiB | 5,220,492 KiB | 26.6 / 26.6 °C |

La primera solicitud ya se ejecutó con el modelo residente, por lo que los
14.89 s no incluyen la carga inicial del servidor. Las solicitudes 2 y 3
bajaron a una media de aproximadamente 4.04 s.

Frente al process-cold medio de 23.145 s, el mejor caso warm estable observado
reduce la latencia aproximadamente un 82.5 %. El RSS permanece alrededor de
2.17 GiB y no se observó calentamiento durante esta prueba corta.

**Cautela:** estas tres solicitudes utilizaron exactamente el mismo prompt con
`cache_prompt=true`. Por ello, las solicitudes 2 y 3 representan un mejor
caso de reutilización del prompt/KV cache y no deben interpretarse todavía como
la latencia típica de un contexto que cambia con nuevos eventos.

Antes de cerrar la evaluación de rendimiento se debe medir warm con snapshots
variables y después la concurrencia con FatigueMonitor, cámara y MediaPipe.

## H7.9.5 — Calidad observada en warm dinámico

En la ejecución dinámica se obtuvieron tres respuestas. Una quedó por debajo
del mínimo actual de 10 palabras; las otras dos quedaron dentro del rango
10–24 usado por `SafeContextualMessageComposer`.

Resultado provisional frente al validador actual: 2 de 3 candidatos serían
aceptables por longitud y formato básico. La muestra sigue siendo insuficiente
para una tasa final y aún faltan validaciones semánticas de hechos, snapshot,
contexto personal y evidencia.

También se detectó una brecha de latencia: `LocalLanguageRequest` tiene un
timeout predeterminado de 3500 ms, mientras el warm dinámico observado fue de
10860–12240 ms. No se cambia todavía ese timeout; debe resolverse antes de JNI
con la política asíncrona/cancelable definida para H7.9.

## Arquitectura híbrida

1. el motor determinista emite inmediatamente la alerta y la acción segura;
2. el LLM trabaja fuera del camino crítico;
3. el contexto generado queda asociado a un snapshot del estado;
4. la respuesta se usa únicamente si sigue vigente y supera el validador;
5. una alerta crítica nueva tiene prioridad sobre cualquier comentario del LLM;
6. cualquier timeout, error, OOM o rechazo deja intacta la ruta determinista.

## Fuentes y licencias

- llama.cpp:
  https://github.com/ggml-org/llama.cpp
- Runtime Android ARM64 fijado:
  https://github.com/ggml-org/llama.cpp/releases/tag/b11073
- Modelos GGUF:
  https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF
  https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF
- Qwen3 / Apache 2.0:
  https://huggingface.co/Qwen/Qwen3-0.6B
  https://huggingface.co/Qwen/Qwen3-1.7B
- Android NDK/CMake:
  https://developer.android.com/studio/projects/install-ndk

Antes de distribuir una APK se revisarán nuevamente licencias y atribuciones.
En H7.9 los artefactos se instalan únicamente en
`/data/local/tmp/fatigue-llm`.

## Preparación del 1.7B

```bash
cd ~/fatigue-driver-prototype
git pull --ff-only

cd android/FatigueMonitor

MODEL_PRESET=1.7b-q4_k_m \
bash scripts/h7_9_prepare_llm_android.sh
```

El script valida ADB/ARM64/API, descarga el runtime fijado, descarga el modelo,
calcula SHA-256 y copia temporalmente los artefactos al teléfono.

## Benchmark H7.9.2 preservado

El benchmark original permanece disponible sin cambios:

```bash
EXPECTED_MODEL_PRESET=1.7b-q4_k_m \
REPEATS=3 \
bash scripts/h7_9_benchmark_llm_android.sh
```

Sus repeticiones crean procesos nuevos; por tanto se consideran
**process-cold** y no warm real.

## Benchmark extendido H7.9.4/H7.9.5

Nuevo script:

```text
scripts/h7_9_benchmark_llm_extended.sh
```

### Process-cold, cinco repeticiones

```bash
MODE=process-cold \
THREADS_LIST="4 6" \
TEMPERATURES="0.0 0.3 0.7" \
REPEATS=5 \
bash scripts/h7_9_benchmark_llm_extended.sh
```

### Warm real

```bash
MODE=warm-server \
THREADS_LIST="4 6" \
TEMPERATURES="0.0 0.3 0.7" \
REPEATS=5 \
bash scripts/h7_9_benchmark_llm_extended.sh
```

El modo warm mantiene `llama-server` cargado y realiza solicitudes
consecutivas al mismo proceso. Si el paquete fijado no contiene
`llama-server`, la fase queda marcada como `SKIPPED`; no se sustituye por
una falsa medición warm.

### Concurrencia con FatigueMonitor + cámara + MediaPipe

Primero abre la aplicación en el teléfono, deja la cámara frontal visible y
mantén la pantalla desbloqueada.

Después ejecuta:

```bash
MODE=all \
CONCURRENT_APP=1 \
THREADS_LIST="4 6" \
TEMPERATURES="0.0 0.3 0.7" \
REPEATS=5 \
bash scripts/h7_9_benchmark_llm_extended.sh
```

También se puede pedir al script que lance la actividad:

```bash
MODE=all \
CONCURRENT_APP=1 \
LAUNCH_APP=1 \
REPEATS=5 \
bash scripts/h7_9_benchmark_llm_extended.sh
```

La vista de cámara debe permanecer visible durante toda la medición.

## Evidencia registrada por el benchmark extendido

Por ejecución se guardan:

- tiempo completo;
- peak RSS del LLM;
- peak RSS de la APK;
- mínimo `MemAvailable`;
- mínimo `SwapFree`;
- temperatura inicial/final;
- respuesta completa;
- código de salida.

Además se conservan:

- `boot_id`;
- modelo/preset;
- snapshot de memoria;
- `logcat` posterior;
- `dumpsys meminfo` de la APK.

Los resultados quedan en:

```text
android/FatigueMonitor/.local/h7_9_llm/benchmarks_extended/<RUN_ID>/
```

El script detiene nuevas pruebas cuando la batería alcanza 42 °C.

## H7.9.6 — Concurrencia real con FatigueMonitor + MediaPipe

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T184456Z
MODE=warm-server
WARM_CONTEXT_MODE=dynamic
CONCURRENT_APP=1
THREADS=6
TEMPERATURE=0.3
REPEATS=3
```

Observación manual del dispositivo: la aplicación se percibió ligeramente más
lenta, pero no se congeló, no se cerró y la degradación no afectó de forma
importante el uso durante esta prueba corta.

Resultados:

| Solicitud | Latencia host | Pico RSS LLM | Pico RSS APK | MemAvailable mínimo | SwapFree mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 21.810 s | 2,275,288 KiB | 245,272 KiB | 3,351,612 KiB | 5,294,220 KiB | 29.3 / 29.3 °C |
| 2 | 34.850 s | 2,306,296 KiB | 252,912 KiB | 3,299,724 KiB | 5,294,220 KiB | 29.3 / 30.6 °C |
| 3 | 25.810 s | 2,339,816 KiB | 247,144 KiB | 3,268,864 KiB | 5,294,988 KiB | 30.6 / 30.6 °C |

Resumen:

- latencia media concurrente: aproximadamente 27.49 s;
- warm dinámico sin APK: aproximadamente 11.68 s;
- degradación de latencia del LLM bajo concurrencia: aproximadamente 2.35x;
- pico RSS de la APK: aproximadamente 247 MiB;
- pico RSS del LLM: aproximadamente 2.23 GiB;
- MemAvailable mínimo observado: aproximadamente 3.12 GiB;
- swap prácticamente sin cambios;
- incremento térmico máximo observado en la secuencia: 1.3 °C;
- sin crash, ANR observable ni congelamiento de la aplicación.

Interpretación provisional: el VIVO Y20 mantiene suficiente margen de memoria
para ejecutar FatigueMonitor y Qwen3-1.7B simultáneamente. El principal
problema observado es la contención de CPU: la latencia del LLM aumenta de
aproximadamente 10–12 s a 22–35 s mientras cámara y MediaPipe están activos.

Antes de descartar el 1.7B se debe comparar concurrencia con menos hilos del
LLM. Reducir de 6 a 4 hilos puede liberar CPU para CameraX/MediaPipe y mejorar
la respuesta percibida de la aplicación, aunque la generación aislada sea algo
más lenta.

## H7.9.6 — Concurrencia con MediaPipe GPU + Qwen CPU (4 hilos)

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T184937Z
MODE=warm-server
WARM_CONTEXT_MODE=dynamic
CONCURRENT_APP=1
MEDIAPIPE_DELEGATE=GPU (seleccionado manualmente en la app)
THREADS=4
TEMPERATURE=0.3
REPEATS=3
```

Resultados:

| Solicitud | Latencia host | Pico RSS LLM | Pico RSS APK | MemAvailable mínimo | SwapFree mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 26.520 s | 2,275,592 KiB | 273,184 KiB | 3,432,480 KiB | 5,296,012 KiB | 33.1 / 33.1 °C |
| 2 | 29.660 s | 2,306,476 KiB | 278,424 KiB | 3,354,948 KiB | 5,293,964 KiB | 33.1 / 33.9 °C |
| 3 | 25.230 s | 2,339,024 KiB | 271,252 KiB | 3,364,476 KiB | 5,293,452 KiB | 33.9 / 33.9 °C |

Resumen:

- latencia media: aproximadamente 27.14 s;
- pico RSS LLM: aproximadamente 2.23 GiB;
- pico RSS APK: aproximadamente 272 MiB;
- MemAvailable mínimo observado: aproximadamente 3.20 GiB;
- swap sin presión apreciable;
- sin fallo de ejecución.

Este resultado no puede atribuirse todavía al delegate GPU porque, respecto de
la línea base previa, cambiaron simultáneamente dos variables: MediaPipe pasó
de CPU a GPU y Qwen pasó de 6 a 4 hilos. La latencia media quedó prácticamente
igual a la línea base CPU+CPU de 6 hilos (27.49 s frente a 27.14 s).

Para aislar el efecto del delegate GPU se debe repetir con MediaPipe GPU y Qwen
en 6 hilos, manteniendo todas las demás condiciones iguales a la línea base.
Después, si se desea aislar el efecto del número de hilos, se debe comparar
CPU+4 frente a GPU+4.

## H7.9.6 — Concurrencia con MediaPipe GPU + Qwen CPU (6 hilos)

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T185257Z
MODE=warm-server
WARM_CONTEXT_MODE=dynamic
CONCURRENT_APP=1
MEDIAPIPE_DELEGATE=GPU (seleccionado manualmente en la app)
THREADS=6
TEMPERATURE=0.3
REPEATS=3
```

Resultados:

| Solicitud | Latencia host | Pico RSS LLM | Pico RSS APK | MemAvailable mínimo | SwapFree mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 24.680 s | 2,274,800 KiB | 279,424 KiB | 3,420,704 KiB | 5,293,708 KiB | 34.7 / 34.7 °C |
| 2 | 36.980 s | 2,305,704 KiB | 280,092 KiB | 3,390,908 KiB | 5,284,236 KiB | 34.7 / 35.1 °C |
| 3 | 32.840 s | 2,339,228 KiB | 283,408 KiB | 3,362,748 KiB | 5,276,880 KiB | 35.1 / 35.7 °C |

Resumen:

- latencia media: 31.50 s;
- pico RSS LLM: aproximadamente 2.23 GiB;
- pico RSS APK: aproximadamente 277 MiB;
- MemAvailable mínimo observado: aproximadamente 3.21 GiB;
- sin error de ejecución.

Comparación provisional:

| Configuración | Latencia media |
|---|---:|
| MediaPipe CPU + Qwen 6 hilos | 27.49 s |
| MediaPipe GPU + Qwen 4 hilos | 27.14 s |
| MediaPipe GPU + Qwen 6 hilos | 31.50 s |

GPU + 6 hilos resultó aproximadamente 14.6 % más lento que la línea base
CPU + 6 hilos y aproximadamente 16.1 % más lento que GPU + 4 hilos.

Sin embargo, esta comparación todavía está afectada por temperatura: la línea
base CPU + 6 comenzó en 29.3 °C, GPU + 4 comenzó en 33.1 °C y GPU + 6 comenzó
en 34.7 °C. La última corrida alcanzó 35.7 °C, por lo que no se puede separar
todavía el efecto del delegate GPU del posible throttling térmico.

Antes de decidir el delegate se requiere una repetición controlada después de
enfriar el dispositivo y, preferiblemente, telemetría explícita del tiempo de
inferencia de MediaPipe para comparar CPU y GPU directamente.

## H7.9.7 — Candidato LFM2.5-1.2B-Instruct

Se añade como candidato principal intermedio:

```text
MODEL_PRESET=lfm2.5-1.2b-q4_k_m
MODEL_FAMILY=lfm2.5
MODEL_FILE=LFM2.5-1.2B-Instruct-Q4_K_M.gguf
```

El GGUF oficial Q4_K_M de Liquid AI ocupa aproximadamente 731 MB. El preset
verifica además el SHA-256 publicado para evitar comparar un artefacto
diferente.

El benchmark conserva los mismos hechos, escenarios y reglas de seguridad que
Qwen, pero adapta el formato del prompt a LFM2.5:

- token inicial `<|startoftext|>`;
- ChatML-like `<|im_start|>...`;
- sin `/no_think` ni bloque `<think>`;
- top-k predeterminado 50;
- repeat penalty predeterminado 1.05.

La comparación funcional se hará en este orden:

1. carga y process-cold mínima;
2. warm dinámico sin APK;
3. inspección de las tres respuestas;
4. concurrencia con FatigueMonitor;
5. comparación de calidad, latencia, RAM y temperatura contra Qwen3-1.7B.

### Preparar LFM2.5

```bash
MODEL_PRESET=lfm2.5-1.2b-q4_k_m \
bash scripts/h7_9_prepare_llm_android.sh
```

### Smoke test inicial

```bash
EXPECTED_MODEL_PRESET=lfm2.5-1.2b-q4_k_m \
MODE=process-cold \
THREADS_LIST="6" \
TEMPERATURES="0.1" \
REPEATS=1 \
bash scripts/h7_9_benchmark_llm_extended.sh
```

No se ejecutará todavía la batería completa hasta confirmar que el runtime
Android carga el modelo correctamente y que la primera respuesta tiene una
calidad lingüística razonable.

## H7.9.7 — Primer benchmark process-cold de LFM2.5-1.2B

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T191758Z
MODEL=LFM2.5-1.2B-Instruct-Q4_K_M.gguf
MODE=process-cold
THREADS=6
TEMPERATURE=0.1
REPEATS=1
CONCURRENT_APP=0
```

Resultados:

- tiempo host: 13.410 s;
- prompt eval: 7.070 s para 256 tokens;
- prompt throughput: 36.21 tok/s;
- generación: 1.446 s para 16 runs;
- generación: 11.07 tok/s;
- tiempo interno total: 8.551 s;
- pico RSS: 1,493,312 KiB;
- MemAvailable mínimo: 4,218,348 KiB;
- SwapFree mínimo: 5,277,648 KiB;
- temperatura: 27.9 / 27.9 °C;
- código de salida: 0.

Comparado con el process-cold previo de Qwen3-1.7B (media de 23.145 s,
aproximadamente 5.98 tok/s y pico RSS medio de aproximadamente 2,319,198 KiB),
LFM2.5 muestra en esta primera ejecución una reducción aproximada del 42 % en
tiempo host, cerca del doble de throughput de generación y alrededor de un
36 % menos de RSS.

Estos valores son preliminares porque solo existe una repetición. El siguiente
paso es medir warm dinámico con los mismos tres snapshots usados para Qwen3-1.7B
y revisar las respuestas generadas antes de probar concurrencia con la APK.

### Corrección de formato LFM2.5 tras el primer smoke test

El primer process-cold mostró el warning de llama.cpp:

```text
Added a BOS token to the prompt ... prompt also starts with a BOS token
```

La salida fue:

```text
¿Qué señales aparecen? El cierre ocular se repitió.
```

Esta salida sería rechazada por el validador actual por formato y longitud.
Sin embargo, no se usa todavía para juzgar la calidad del modelo porque el
prompt contenía dos BOS consecutivos.

La plantilla oficial de LFM2.5 incluye el BOS al ser aplicada por el
tokenizador, pero el flujo raw de llama.cpp usado en este benchmark ya añade
el BOS según los metadatos del modelo. Por ello, el benchmark conserva
`<|im_start|>...` pero elimina el `<|startoftext|>` explícito. Se debe repetir
una sola inferencia process-cold y confirmar que desaparece el warning antes
de ejecutar warm dinámico.

## H7.9.8 — Selector contextual + renderer determinista

A partir de la prueba de LFM2.5 se cambia el objetivo principal del modelo
pequeño: deja de redactar libremente el mensaje y pasa a seleccionar una
combinación cerrada de contexto.

Flujo:

```text
ContextSnapshot
    -> LFM2.5 selector
    -> JSON restringido
    -> ContextualTemplateRenderer
    -> SafeContextualMessageComposer
    -> acción determinista
    -> TTS
```

Ejemplo de salida permitida del modelo:

```json
{
  "template": "repeated_primary_with_recent_events",
  "primary": "eye_closure",
  "secondary": "yawns_and_nod"
}
```

El modelo no puede escribir el texto destinado al usuario. El renderer Kotlin
contiene únicamente combinaciones aprobadas. Para el ejemplo anterior genera:

```text
Joaquín, el cierre ocular prolongado volvió a presentarse; además, se
registraron bostezos y un cabeceo recientemente.
```

Si los IDs son válidos pero la combinación no está aprobada, el renderer
devuelve null y la ruta debe usar el fallback determinista.

Archivos:

```text
app/src/main/java/.../ContextualTemplateRenderer.kt
app/src/test/java/.../ContextualTemplateRendererTest.kt
scripts/h7_9_template_selector.py
```

El benchmark warm puede activar esta arquitectura mediante
`OUTPUT_MODE=template-selector`. La petición a llama-server utiliza
`json_schema`, soportado por el runtime llama.cpp fijado b11073, para impedir
que la decodificación salga del vocabulario estructurado.

### Primera prueba del selector con LFM2.5

```bash
EXPECTED_MODEL_PRESET=lfm2.5-1.2b-q4_k_m \
MODE=warm-server \
WARM_CONTEXT_MODE=dynamic \
OUTPUT_MODE=template-selector \
THREADS_LIST="6" \
TEMPERATURES="0.1" \
REPEATS=3 \
bash scripts/h7_9_benchmark_llm_extended.sh
```

Criterios de aceptación inicial:

- los tres JSON deben poder parsearse;
- las tres combinaciones deben estar en el catálogo aprobado;
- los textos renderizados deben coincidir con hechos presentes en cada
  snapshot;
- ninguna salida del LLM contiene prosa libre;
- se comparará la latencia contra el warm dinámico de generación libre.

## H7.9.8 — Resultado del primer selector estructurado LFM2.5

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T193706Z
MODEL=LFM2.5-1.2B-Instruct-Q4_K_M.gguf
MODE=warm-server
WARM_CONTEXT_MODE=dynamic
OUTPUT_MODE=template-selector
THREADS=6
TEMPERATURE=0.1
REPEATS=3
CONCURRENT_APP=0
```

El renderer Kotlin compiló y sus pruebas unitarias terminaron con
`BUILD SUCCESSFUL`.

Resultados del selector:

1. cierre ocular repetido + bostezos/cabeceo: correcto;
2. bostezos + mirada desviada + cambio respecto al historial: correcto;
3. recuperación después de episodio crítico: incorrecto; el modelo volvió a
   seleccionar la plantilla del caso 1.

La restricción por `json_schema` funcionó: las tres salidas fueron JSON
válidos y pertenecieron al vocabulario permitido. Sin embargo, restringir la
forma no garantiza que el modelo elija la semántica correcta.

Latencia:

| Solicitud | Latencia host | Pico RSS LLM | MemAvailable mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|
| 1 | 14.270 s | 1,507,608 KiB | 4,169,624 KiB | 26.7 / 26.8 °C |
| 2 | 12.860 s | 1,512,532 KiB | 4,186,356 KiB | 26.8 / 27.2 °C |
| 3 | 12.440 s | 1,518,876 KiB | 4,188,432 KiB | 27.2 / 27.2 °C |

Media warm estructurada: aproximadamente 13.19 s. El RSS se mantiene alrededor
de 1.44–1.45 GiB y no se observa calentamiento relevante.

Conclusión arquitectónica: el modelo pequeño no debe seleccionar el significado
principal de la alerta. Estado, trigger, hechos permitidos y familia de
plantilla deben decidirse de forma determinista a partir de FatigueEngine,
ContextualAlertPlanner e historial. El LLM, si se conserva, solo podrá elegir
variaciones estilísticas previamente aprobadas dentro de una familia semántica
ya fijada, o seleccionar qué hecho opcional autorizado enfatizar sin poder
cambiar el significado.

Esto elimina el error observado en recuperación y conserva la ventaja de
plantillas naturales, auditables y multilingües.

## H7.9.8 — Revisión: semántica determinista + estilo LFM

El primer selector estructurado demostró que un `json_schema` garantiza forma
válida, pero no garantiza la selección semántica correcta: el escenario de
recuperación fue clasificado como cierre ocular repetido.

Por ello se restringe aún más el papel del LLM:

1. FatigueEngine + ContextualAlertPlanner + historial fijan la familia
   semántica;
2. el LLM no puede cambiar trigger, estado, hechos ni acción;
3. LFM2.5 solo selecciona `{"variant":"a|b|c"}`;
4. cada variante corresponde a una frase completa previamente revisada;
5. el renderer Kotlin conserva la misma semántica en las tres variantes;
6. cualquier error mantiene disponible el fallback determinista.

Ejemplo para `repeated_primary_with_recent_events`:

- A: "Joaquín, el cierre ocular prolongado volvió a presentarse; además, se
  registraron bostezos y un cabeceo recientemente."
- B: "Joaquín, volvió a registrarse un cierre ocular prolongado, junto con
  bostezos y un cabeceo en los eventos recientes."
- C: "Joaquín, se repitió el cierre ocular prolongado y también se registraron
  bostezos y un cabeceo recientemente."

La nueva prueba usa `OUTPUT_MODE=style-selector` y limita la generación a un
máximo efectivo de 16 tokens.

```bash
EXPECTED_MODEL_PRESET=lfm2.5-1.2b-q4_k_m \
MODE=warm-server \
WARM_CONTEXT_MODE=dynamic \
OUTPUT_MODE=style-selector \
THREADS_LIST="6" \
TEMPERATURES="0.1" \
REPEATS=3 \
bash scripts/h7_9_benchmark_llm_extended.sh
```

El objetivo de esta prueba ya no es medir comprensión semántica del LLM, sino
su coste como selector ligero de estilo sobre una decisión segura ya cerrada.

## H7.9.8 — Resultado del selector de estilo LFM2.5

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T194325Z
MODEL=LFM2.5-1.2B-Instruct-Q4_K_M.gguf
MODE=warm-server
WARM_CONTEXT_MODE=dynamic
OUTPUT_MODE=style-selector
THREADS=6
TEMPERATURE=0.1
REPEATS=3
CONCURRENT_APP=0
```

Las tres selecciones fueron aceptadas y mantuvieron la semántica fijada por el
motor determinista:

1. cierre ocular repetido + bostezos/cabeceo -> variante B;
2. bostezos + desviación de mirada + cambio respecto al historial -> variante B;
3. recuperación después de episodio crítico -> variante B.

Textos renderizados:

```text
Joaquín, volvió a registrarse un cierre ocular prolongado, junto con bostezos
y un cabeceo en los eventos recientes.

Joaquín, ahora se registran bostezos repetidos y una desviación prolongada de
la mirada, a diferencia de las señales leves previas.

Joaquín, se observa una recuperación sostenida tras el episodio crítico
previo, sin nuevos bostezos ni cabeceos.
```

Rendimiento:

| Solicitud | Latencia host | Pico RSS LLM | MemAvailable mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|
| 1 | 7.080 s | 1,497,536 KiB | 4,224,748 KiB | 27.4 / 27.4 °C |
| 2 | 7.250 s | 1,501,160 KiB | 4,222,728 KiB | 27.4 / 27.4 °C |
| 3 | 7.150 s | 1,504,188 KiB | 4,226,420 KiB | 27.4 / 27.4 °C |

Media warm: aproximadamente 7.16 s. El rango entre las tres solicitudes es de
solo 170 ms, por lo que la latencia fue estable en esta prueba corta. El RSS se
mantuvo alrededor de 1.43 GiB y no se observó incremento térmico.

Frente al selector semántico anterior (media aproximada de 13.19 s), delegar
solo el estilo redujo la latencia alrededor de un 45.7 %. Frente al warm
dinámico de Qwen3-1.7B con generación libre (media aproximada de 11.68 s),
esta configuración es alrededor de un 38.7 % más rápida, además de utilizar
considerablemente menos memoria y eliminar la generación libre de la ruta
contextual.

Este resultado convierte a LFM2.5 + semántica determinista + renderer de
plantillas en la configuración principal candidata para la siguiente prueba de
concurrencia con FatigueMonitor.

## H7.9.8 — LFM2.5 concurrente con FatigueMonitor

Ejecución reportada el 24 de septiembre de 2026:

```text
RUN_ID=20260924T194641Z
MODEL=LFM2.5-1.2B-Instruct-Q4_K_M.gguf
MODE=warm-server
WARM_CONTEXT_MODE=dynamic
OUTPUT_MODE=style-selector
THREADS=6
TEMPERATURE=0.1
REPEATS=3
CONCURRENT_APP=1
```

La APK fue detectada activa con la vista de cámara durante la prueba. Las tres
salidas fueron aceptadas y renderizadas con la semántica ya fijada por la
lógica determinista.

| Solicitud | Latencia host | Pico RSS LLM | Pico RSS APK | MemAvailable mínimo | Temp. inicial/final |
|---|---:|---:|---:|---:|---:|
| 1 | 11.890 s | 1,500,596 KiB | 238,904 KiB | 3,987,884 KiB | 28.3 / 28.3 °C |
| 2 | 11.740 s | 1,500,684 KiB | 239,000 KiB | 3,976,804 KiB | 28.3 / 28.3 °C |
| 3 | 11.670 s | 1,503,700 KiB | 236,916 KiB | 3,978,120 KiB | 28.3 / 28.3 °C |

Media concurrente: aproximadamente 11.77 s. Frente al mismo selector de estilo
sin APK (media aproximada de 7.16 s), la concurrencia incrementó la latencia en
aproximadamente 64 %. Aun así, sigue siendo muy inferior al resultado previo de
Qwen3-1.7B concurrente con MediaPipe GPU y 6 hilos (media aproximada de 31.50 s).

El RSS del LLM permanece alrededor de 1.43 GiB; el RSS de la APK fue de
aproximadamente 231–233 MiB. La temperatura de batería permaneció en 28.3 °C
durante las tres solicitudes y no hubo errores de ejecución.

Nota metodológica: esta ejecución confirma concurrencia con la APK, pero el
benchmark todavía no registra automáticamente el delegate efectivo de
MediaPipe ni su latencia por frame. Esos dos campos deben medirse directamente
antes de afirmar en el artículo una comparación CPU vs GPU.

## H7.9.9 — Telemetría directa de MediaPipe

Se instrumenta la ruta LIVE_STREAM para separar tres métricas por callback:

- `preprocessing_ms`: desde la recepción del frame hasta el envío a
  `detectAsync`; incluye copia RGBA, rotación/reflejo y construcción MPImage.
- `mediapipe_ms`: desde el envío a `detectAsync` hasta el callback de
  Face Landmarker.
- `end_to_end_ms`: suma observable desde el timestamp inicial del frame hasta
  el callback.

También se registra el delegate que quedó realmente activo después de crear
Face Landmarker: `CPU`, `GPU` o `UNKNOWN`.

Para reducir perturbación, no se escribe una línea por frame. La app acumula
ventanas no solapadas de 30 callbacks y registra por ventana:

- media, p50, p95 y máximo de end-to-end;
- media, p50, p95 y máximo de preprocesamiento;
- media, p50, p95 y máximo de MediaPipe;
- FPS efectivo de callbacks;
- delegate efectivo;
- uptime de inicio y fin de ventana.

El benchmark limpia logcat al comenzar y guarda el uptime del dispositivo.
El parser descarta cualquier ventana que haya empezado antes del benchmark,
evitando contaminar los resultados con frames previos.

Cada RUN_ID nuevo puede generar:

```text
mediapipe_measurement_start_uptime_ms.txt
mediapipe_telemetry.tsv
logcat_after.txt
```

El TSV permite verificar automáticamente el delegate y comparar CPU vs GPU
sin depender de la selección manual de la interfaz.

## H7.9.9 — Protocolo CPU vs GPU de MediaPipe

Antes de volver a combinar MediaPipe con LFM2.5 se ejecuta un control aislado
sin LLM mediante `h7_9_benchmark_mediapipe_only.sh`.

Condiciones recomendadas:

- misma cámara frontal;
- rostro visible en posición comparable;
- mismo intervalo de análisis de 100 ms;
- 30 s por condición;
- temperatura inicial lo más parecida posible;
- primero una condición y luego la otra, documentando el orden;
- usar únicamente resultados donde el delegate verificado coincida con el
  esperado.

CPU:

```bash
EXPECTED_DELEGATE=CPU SAMPLE_DURATION_S=30 \
bash scripts/h7_9_benchmark_mediapipe_only.sh
```

GPU:

```bash
EXPECTED_DELEGATE=GPU SAMPLE_DURATION_S=30 \
bash scripts/h7_9_benchmark_mediapipe_only.sh
```

Cada ejecución produce un `mediapipe_telemetry.tsv` con ventanas de 30
callbacks. Después de comparar CPU y GPU se repetirá LFM2.5 + APK con la opción
seleccionada y con 4/6 hilos para evaluar contención.

### Corrección de reloj en H7.9.9

La primera ejecución del benchmark MediaPipe-only no produjo ventanas válidas.
La causa identificada fue una comparación entre dos relojes distintos:
`SystemClock.uptimeMillis()` en la app y `/proc/uptime` en el script. El
primero excluye tiempo de suspensión, mientras el segundo representa tiempo
desde boot; por ello el filtro temporal podía descartar ventanas válidas.

La telemetría de ventanas usa ahora `SystemClock.elapsedRealtime()`, que se
alinea con el criterio de tiempo desde boot usado por el host. El benchmark
también reporta cuántas líneas `H7_9_PERF` crudas encontró cuando el parser
no produce ventanas, para distinguir entre APK no instrumentada y un problema
de filtrado.

## H7.9.9 — Matriz concurrente final propuesta

Tras los baselines aislados:

- CPU: 77.05 ms E2E medio, 63.39 ms MediaPipe medio, 8.83 FPS;
- GPU: 91.08 ms E2E medio, 63.00 ms MediaPipe medio, 8.63 FPS.

GPU no mejora la media de MediaPipe en aislamiento, pero reduce claramente la
latencia de cola. CPU tiene mejor E2E medio. Por ello la selección final no se
realiza todavía: debe medirse la contención con LFM2.5.

Matriz concurrente:

1. MediaPipe CPU + LFM2.5 4 hilos;
2. MediaPipe CPU + LFM2.5 6 hilos;
3. MediaPipe GPU + LFM2.5 4 hilos;
4. MediaPipe GPU + LFM2.5 6 hilos.

Cada ejecución usa `OUTPUT_MODE=style-selector`, contexto dinámico, tres
repeticiones y verifica automáticamente el delegate efectivo mediante
`EXPECTED_MEDIAPIPE_DELEGATE`.

La decisión final debe considerar conjuntamente latencia LLM, E2E MediaPipe,
p95 MediaPipe, callback FPS, memoria y estado térmico.

## Lo que aún debe medirse antes de JNI

1. varias inferencias process-cold del 1.7B;
2. warm real con el modelo residente;
3. estabilidad de latencia y temperatura;
4. calidad con distintas temperaturas de sampling;
5. secuencias con memoria de eventos;
6. concurrencia con cámara y MediaPipe;
7. presión de memoria de APK + LLM;
8. impacto sobre la latencia de MediaPipe;
9. prioridad de la alerta fija frente al comentario contextual;
10. tasa de respuestas aceptadas/rechazadas por el validador.

La telemetría específica de tiempo de MediaPipe se añadirá antes de la decisión
final del modelo. H7.10/JNI permanece bloqueado hasta completar estas pruebas.
