# H7.9.4 — Arquitectura contextual y protocolo de evaluación del LLM local

## Objetivo corregido de H7.9

H7.9 no convierte al LLM en el motor de seguridad ni sustituye las alertas
deterministas ya implementadas en H7.7 y H7.8.

El objetivo es añadir, de forma completamente local y desacoplada del camino
crítico, un asistente de voz capaz de comentar la evolución de la fatiga a
partir de:

- señales observadas: EAR, PERCLOS, MAR, gaze y head pose;
- historial de eventos de la sesión;
- memoria de lo que el propio asistente ya comunicó;
- contexto personal explícitamente autorizado;
- evidencia científica local previamente curada.

La salida del LLM nunca puede modificar el estado de fatiga, la urgencia, la
acción obligatoria ni los tiempos de repetición definidos por el sistema
determinista.

## Lo que ya existe y se conserva

No se reemplazan las implementaciones validadas:

| Componente | Archivo principal | Función |
|---|---|---|
| Percepción facial | `FaceLandmarkerHelper.kt`, `FatigueCues.kt` | Landmarks y señales geométricas |
| Temporales | `FatigueTemporal.kt`, `HeadPoseTemporal.kt` | PERCLOS, bostezos, mirada, pose |
| Fusión | `FatigueEngine.kt` | Estado y razones |
| Historial local | `FatigueHistory.kt` | Persistencia de eventos |
| Política de alerta | `ContextualAlerts.kt` | Urgencia, intención y acción fija |
| Validador lingüístico | `SafeContextualLanguage.kt` | Rechazo y fallback |
| Voz offline | `AndroidOfflineAlertSpeaker.kt` | TTS, cola e interrupción |
| Integración cámara | `fragment/CameraFragment.kt` | Orquestación existente |

En el estado actual, `CameraFragment` envía el `fallbackMessage`
determinista directamente al TTS. Ese comportamiento permanece como ruta
segura y no depende del LLM.

## Arquitectura objetivo

### Camino crítico de seguridad

```text
CameraX + MediaPipe
        |
        v
EAR / PERCLOS / MAR / gaze / head pose
        |
        v
FatigueEngine
        |
        v
ContextualAlertPlanner
        |
        +----> alerta determinista inmediata ----> AndroidOfflineAlertSpeaker
```

Este camino debe funcionar aunque:

- el LLM no exista;
- el modelo no pueda cargarse;
- Android tenga presión de memoria;
- la generación expire;
- la respuesta sea inválida;
- la respuesta llegue tarde.

### Camino contextual no crítico

```text
ContextSnapshot
  + FatigueHistory
  + AssistantMemory
  + AuthorizedPersonalContext
  + EvidenceCards
        |
        v
PromptBuilder
        |
        v
LLM local
        |
        v
StrictContextValidator
        |
        +----> válido y vigente ----> SpeechCoordinator ----> TTS
        |
        +----> inválido/obsoleto ----> descartar
```

El comentario contextual se genera después de que la alerta determinista haya
sido emitida. Una nueva alerta crítica puede cancelar cualquier comentario
contextual pendiente.

## Contratos de datos propuestos

Estos contratos documentan el diseño. No se implementan todavía en JNI.

### ContextSnapshot

Debe identificar de forma inmutable el estado que originó una solicitud al
LLM.

Campos mínimos:

```text
snapshot_id
created_at_ms
state
urgency
intent
trigger
score
closure_s
perclos
yawns_5m
nods_2m
gaze_away_count
pose_away_count
reasons[]
history_summary
mandatory_action_id
```

La respuesta del LLM se acepta únicamente si su `snapshot_id` sigue vigente.

### AssistantMemoryEntry

Evita repetir literalmente hechos o recomendaciones ya pronunciadas.

```text
message_id
timestamp_ms
snapshot_id
facts_used[]
evidence_ids[]
message_text
delivery_status
```

La memoria del asistente es distinta del historial biométrico. El historial
describe lo observado; la memoria describe lo que ya se comunicó.

### AuthorizedPersonalContext

Los datos personales no se infieren y no se activan por defecto.

```text
context_id
type
value
unit
recorded_at_ms
source
authorized_for_assistant
expires_at_ms
```

Ejemplos admisibles si el usuario los autorizó: duración del sueño, hora de
inicio del sueño, ejercicio registrado o descanso reciente.

Reglas:

1. un dato sin autorización no entra al prompt;
2. un dato vencido no entra al prompt;
3. el modelo no puede transformar correlaciones en causalidad;
4. el contexto personal no modifica por sí solo el nivel de fatiga.

### EvidenceCard

La evidencia científica debe ser local, versionada y seleccionada antes de
invocar al LLM.

```text
evidence_id
topic
claim
recommendation
conditions[]
source_title
source_year
doi_or_url
version
```

El LLM puede redactar una recomendación científica únicamente cuando el prompt
incluya al menos un `evidence_id` aplicable. El modelo no inventa fuentes.

## Prioridades de voz

Orden de prioridad:

1. alerta crítica determinista;
2. alerta determinista no crítica;
3. reconocimiento o recuperación;
4. comentario contextual del LLM.

Una alerta crítica nueva debe interrumpir cualquier comentario contextual. Un
comentario contextual nunca interrumpe una alerta crítica.

## Validador contextual objetivo

`SafeContextualLanguage.kt` se conserva como base. La evolución posterior
debe validar, como mínimo:

- longitud y formato;
- ausencia de tokens de plantilla;
- ausencia de diagnóstico o medicación;
- ausencia de culpa o moralización;
- ausencia de la acción obligatoria reservada;
- presencia de señales realmente observadas;
- correspondencia con el `snapshot_id`;
- contexto personal autorizado;
- recomendación científica respaldada por `evidence_id`;
- ausencia de hechos inventados;
- ausencia de contradicción con el estado actual;
- no repetición excesiva frente a `AssistantMemory`.

Si falla cualquier regla, el comentario contextual se descarta. La alerta
determinista ya habrá sido pronunciada.

## Resultados conocidos antes de H7.9.4

### Qwen3-0.6B Q4_0

- tiempo completo observado: 7.77 s;
- pico aproximado: 850 MB;
- calidad lingüística insuficiente en las pruebas realizadas.

### Qwen3-1.7B Q4_K_M

- una inferencia documentada;
- 27.16 s de tiempo completo en frío;
- 16.91 s de inferencia interna;
- 6.09 tokens/s;
- aproximadamente 2.27 GiB de pico RSS;
- mejor calidad lingüística que 0.6B;
- la respuesta repitió la acción obligatoria y fue rechazada.

Una inferencia no es suficiente para aceptar o descartar el 1.7B.

## Terminología de rendimiento

Para evitar conclusiones incorrectas:

- **process-cold**: se crea un proceso nuevo y el modelo se vuelve a cargar;
- **device-cold**: primera ejecución después de reiniciar el teléfono;
- **warm real**: el mismo proceso conserva el modelo cargado y atiende varias
  solicitudes;
- **concurrent**: la APK permanece en primer plano con cámara y MediaPipe
  activos mientras el LLM genera.

Ejecutar varias veces `llama-cli` no constituye una prueba warm real aunque
el kernel mantenga páginas en caché.

## Protocolo de evaluación del Qwen3-1.7B

### Fase A — process-cold reproducible

Configuración inicial:

- Qwen3-1.7B Q4_K_M;
- contexto 1024;
- salida máxima 48 tokens;
- CPU;
- 4 y 6 hilos;
- semilla fija;
- GPU desactivada.

Realizar al menos cinco repeticiones por configuración y registrar:

- tiempo total del host;
- prompt eval;
- eval;
- tokens/s;
- peak RSS del LLM;
- mínimo `MemAvailable`;
- mínimo `SwapFree`;
- temperatura antes y después;
- texto generado;
- código de salida.

Para una medición **device-cold**, reiniciar manualmente el teléfono y ejecutar
una sola repetición. El benchmark registra el `boot_id` para conservar
trazabilidad.

### Fase B — warm real

Si el runtime fijado contiene `llama-server`, mantener un único proceso con el
modelo cargado y efectuar varias solicitudes consecutivas.

Medir por solicitud:

- latencia completa;
- RSS del servidor;
- memoria disponible;
- temperatura;
- texto devuelto.

Si `llama-server` no está disponible, el benchmark debe marcar la fase como
`SKIPPED`; no se permite renombrar repeticiones de `llama-cli` como warm.

### Fase C — muestreo y estabilidad lingüística

Comparar al menos:

- temperatura 0.0;
- temperatura 0.3;
- temperatura 0.7.

Usar la misma batería de escenarios y la misma semilla cuando sea aplicable.

Escenarios mínimos:

1. cierre ocular aislado;
2. dos bostezos sucesivos;
3. PERCLOS elevado;
4. cabeceos;
5. mirada apartada;
6. transición SUSPECTED -> FATIGUE -> CRITICAL;
7. recuperación;
8. recuperación seguida de recaída;
9. historial con eventos repetidos;
10. contexto personal deshabilitado;
11. contexto personal autorizado;
12. recomendación con EvidenceCard.

La métrica principal no es creatividad. Es la proporción de respuestas que
pasan el validador sin inventar hechos.

### Fase D — memoria del asistente

Ejecutar una secuencia temporal en una misma sesión:

```text
bostezo
-> segundo bostezo
-> cabeceo
-> cierre ocular prolongado
-> recuperación
-> nueva recurrencia
```

Comprobar que el texto:

- reconoce evolución solo cuando el historial la demuestra;
- no repite literalmente el mensaje anterior;
- no inventa eventos;
- no transforma ejercicio o sueño en causa demostrada;
- no reutiliza recomendaciones científicas fuera de sus condiciones.

### Fase E — concurrencia con la APK

Mantener la aplicación en primer plano, pantalla encendida y cámara activa.
Ejecutar el LLM desde `/data/local/tmp/fatigue-llm` sin JNI.

Registrar:

- RSS del LLM;
- RSS de la APK;
- `MemAvailable`;
- swap;
- temperatura;
- crashes, ANR y eventos del low-memory killer;
- latencia de generación;
- estabilidad de la cámara;
- continuidad de MediaPipe;
- capacidad de emitir la alerta determinista mientras el LLM está ocupado.

El benchmark extendido automatiza memoria, temperatura, logs y latencia del
LLM. La latencia de MediaPipe debe incorporarse como telemetría específica
antes de tomar la decisión final del modelo.

### Fase F — estrés controlado

Con la APK activa, realizar varias solicitudes contextuales sucesivas.

Reglas:

- nunca ejecutar dos instancias del 1.7B simultáneamente;
- una solicitud nueva puede invalidar una respuesta anterior;
- una alerta crítica cancela cualquier comentario contextual pendiente;
- una respuesta que termina con un snapshot obsoleto se descarta;
- no se permite que el LLM retrase la alerta fija.

## Criterios de decisión antes de JNI

JNI permanece bloqueado hasta contar con evidencia de todas las fases
anteriores.

El candidato solo avanza si:

1. no provoca cierre, ANR ni muerte de la APK;
2. no impide el funcionamiento continuo de cámara y MediaPipe;
3. deja margen de memoria reproducible;
4. no presenta degradación térmica sostenida inaceptable;
5. la ruta determinista sigue respondiendo independientemente del LLM;
6. el validador rechaza contenido inseguro u obsoleto;
7. existe una tasa de respuestas válidas suficientemente alta para justificar
   el coste de memoria y latencia;
8. las mediciones warm y concurrentes están documentadas, no inferidas.

No se fija todavía un umbral absoluto de latencia del comentario contextual.
Se definirá con los resultados reales del vivo V2058.

## Hitos siguientes

| Hito | Entregable |
|---|---|
| H7.9.4 | Arquitectura y protocolo fijados |
| H7.9.5 | Cold/warm, repeticiones y sampling |
| H7.9.6 | APK + MediaPipe + LLM concurrentes |
| H7.9.7 | AssistantMemory + contexto autorizado + EvidenceCards |
| H7.9.8 | Validador contextual integral |
| H7.9.9 | Decisión de modelo/configuración |
| H7.10 | JNI solo para el candidato aceptado |
