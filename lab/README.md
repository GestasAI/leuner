# Laboratorio de proteínas

Código, vectores y mediciones de la fase 1 de la dendrita de Launer y de los primeros ensayos de las fases 2 y 4. Todo lo que aquí se afirma tiene un número reproducible con semilla fija.

## Qué es una proteína

Una capacidad condensada en un vector. Se extrae del propio modelo leyendo sus activaciones mientras resuelve una tarea con ejemplos, y se vuelve a inyectar en una capa intermedia cuando ya no hay ejemplos. La técnica procede de los vectores de tarea (Hendel, Geva y Globerson, 2023) y de los vectores de función (Todd et al., ICLR 2024).

1. Se preparan entre 10 y 20 pares entrada → salida, presentados como turnos de conversación (con texto plano, el modelo IT copia el último ejemplo y acierta el 0 %).
2. Se captura el estado oculto del último token en la capa L.
3. La proteína es la media de esos estados: 1.536 dimensiones, 6 KB en float32.
4. En un prompt sin ejemplos se suma `λ·v` en la capa L (`protein_core.inject`).
5. Se compara el acierto sin y con proteína, con semillas fijas.

Modelo: `google/gemma-4-E2B-it` (35 capas). Receptor: capa 20 (≈ 57 % de la profundidad). Escala λ = 4,0. CPU en bfloat16; no hace falta GPU.

## Instalación y uso

```bash
pip install -r requirements.txt
huggingface-cli login          # aceptar la licencia de Gemma en el Hub

python f0_function_vector.py --model google/gemma-4-E2B-it --task capital --layers 20   # ¿existe la proteína?
python f1_protein_probes.py   --model google/gemma-4-E2B-it                              # generación, colateral, composición
python f2_extract_protein.py  --model google/gemma-4-E2B-it --task translate             # extraer y guardar un artefacto
python f4_robustness.py       --model google/gemma-4-E2B-it                              # cinco semillas
python f8_cross_task.py       --model <ruta local del checkpoint>                        # prueba cruzada y proteína mezclada
python f8b_lenient_measure.py --model <ruta local del checkpoint>                        # métrica sobre la respuesta generada
python f9_orchestration.py    --model <ruta local del checkpoint>                        # enrutador y cadena de tres proteínas
```

Los scripts F8 y F9 aceptan una carpeta local con el checkpoint en formato Hugging Face (`config.json`, `model.safetensors`, `tokenizer.json`).

## Artefactos

Cada proteína se guarda como `proteins/<capacidad>.<modelo>.L<capa>.s<escala>.f32` (vector crudo, float32 little-endian, cargable desde Python, Rust o Dart) y un `.json` con metadatos, eficacia medida y SHA-256.

| Artefacto | Capacidad | Acierto | Método |
|---|---|---|---|
| `translate_gemma_4_e2b_it.protein.npz` | inglés → español | 0 % → 90 % | media de activaciones, formato antiguo |
| `translate_es_en.gemma-4-E2B-it.L20.s4.0.f32` | español → inglés | 0 % → 93 % | media de activaciones |
| `capital.gemma-4-e2b-it-hf.L20.s4.0.f32` | país → capital | 0 % → 93 % | media de activaciones |
| `composer_country.gemma-4-e2b-it-hf.L20.s4.0.f32` | compositor → país | 0 % → 80 % | media de activaciones |
| `mixed_capital_es_en.gemma-4-e2b-it-hf.L20.s4.0.f32` | capitales + español → inglés | 93 % y 53 % | extracción de demostraciones mezcladas |

## Resultados

### F0 · Existe la proteína (9 de junio de 2026)

| Modelo | Tarea | Capa | Sin ejemplos | Con proteína | Con ejemplos en el contexto |
|---|---|---|---|---|---|
| Gemma 4 E2B-it | país → capital | 20/35 | 0 % | **90 %** | 95 % |
| Gemma 2B-it (control) | antónimos | 11/18 | 10 % | 80 % | 90 % |

### F1 · Batería (9 de junio)

- Traducción inglés → español, primera proteína propia: 0 % → 90 %.
- Generación completa: 5 de 5 palabras exactas y fluidas; la proteína cambia el modo de responder, de una explicación a la palabra.
- Daño colateral: la proteína de capitales no altera los antónimos (0 puntos).
- Composición por suma: dos proteínas sumadas en el mismo punto se anulan (ambas caen a 0 % en todas las escalas de 0,5 a 6,0). Es dominancia de dirección, no fuga.

### F4 · Robustez (11 de junio)

Traducción inglés → español en cinco semillas: 0,90 / 0,70 / 0,85 / 0,70 / 0,85. Media 0,80 ± 0,09; intervalo del 95 % entre 0,73 y 0,87.

### F5 · Proteínas nuevas (11 de junio)

Español → inglés: 0 % → 93 %. Resumir y clasificar intención no superaron el umbral con la métrica de primer token.

### F8 · Prueba cruzada (22 de septiembre)

¿Activa una proteína capacidades distintas de la suya? Seis tareas, cuatro condiciones, métrica de primer token:

| Tarea | Base | + traducción | + capitales | + mezclada |
|---|---|---|---|---|
| Capitales | 0 % | 0 % | 93 % | 93 % |
| Traducción español → inglés | 0 % | 93 % | 0 % | 0 %* |
| Antónimos, continentes, notas musicales, número siguiente | 0 % | 0 % | 0 % | 0 % |

\* Medido sobre la respuesta generada (palabra esperada en ocho tokens, sin distinguir mayúsculas), la mezclada traduce el 53 % (respuestas como "Love", "Night", "Perro = dog" que el primer token contaba como fallo).

Lo que escribe el modelo: la proteína de capitales ante "Spanish to English: perro =" responde **"Madrid"**; ante "Number after eleven:" responde **"New"**. La de traducción ante "Capital of France:" responde **"France"**. En antónimos y notas es inerte. Conclusión: una proteína no es multitarea; fuera de su tarea es inerte o impone su capacidad. El receptor es común; la proteína, específica.

### F9 · Orquestación (22 de septiembre)

Tres proteínas (compositor → país, capital, traducción) y un enrutador que decide en cada paso cuál inyectar: una elección tipada entre A, B, C o D con una probabilidad por opción, leída de los logits del modelo, sin entrenar.

- Proteína compositor → país: 0 % → 80 %.
- Enrutador: 36 de 42 decisiones correctas (86 %). Los seis fallos, todos al pedir traducir el nombre de una capital. Cuando acierta, la probabilidad es ≈ 1,0; cuando falla, ≈ 0,4–0,5: sobreconfiado, pero la incertidumbre aparece donde falla.

| Condición | País | Capital | Español | Extremo a extremo |
|---|---|---|---|---|
| Orquestada (enrutador + tres proteínas) | 80 % | 87 % | 87 % | **80 %** |
| Base, sin proteínas | 0 % | 0 % | 0 % | 0 % |
| Proteína fija equivocada | 13 % | 0 % | 0 % | 0 % |

Ejemplos: Mozart → Austria → Vienna → Viena; Bizet → France → Paris → París; Tchaikovsky → Russia → Moscow → Moscú. Los fallos son de conocimiento del modelo (Nielsen → Noruega; Liszt → Austria), no del mecanismo.

## Límites

Muestras de 15 a 20 casos y una semilla salvo F4. La métrica de primer token es estricta y oculta aciertos (usar `f8b`). La prueba en cuantización int4 es un sustituto en bfloat16. Ningún runtime móvil probado expone las capas internas. Todas las proteínas medidas son lingüísticas o factuales; ninguna es todavía emocional.

## Informes

`reports/` contiene los JSON con cada medición: `robustness_report.json` (F4), `f5_results.json`, `int4_survival_report.json` (F3), `cross_task_report.json` y `cross_task_lenient_report.json` (F8) y `orchestration_report.json` (F9), con las respuestas generadas caso por caso.
