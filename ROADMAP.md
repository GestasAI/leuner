# Hoja de ruta

Leuner avanza por olas. Solo hay una activa, y una ola se cierra cuando sus hipótesis tienen un resultado medido, positivo o negativo, y publicado. Cada componente "inspirado en el cerebro" tiene que superar su ablación: si retirarlo no cambia nada, no entra.

## Hipótesis contrastables

| Id | Hipótesis | Éxito si |
|---|---|---|
| D1 | La compartimentación (ramas, compuerta competitiva y norma) resuelve la dominancia entre proteínas | Cada proteína conserva ≥ 80 % de su acierto individual con otra activa |
| D2 | Una compuerta de coincidencia reduce activaciones indebidas sin perder eficacia | Menos activaciones indebidas y ≥ 90 % del acierto sin compuerta |
| D3 | La eficacia frente a la escala de inyección tiene un óptimo interior | Óptimo interior en ≥ 3 de 4 tareas |
| D4 | Una proteína extraída de experiencia mezclada activa todas sus capacidades (proteína polivalente) | ≥ 80 % del acierto de cada específica; colateral ≤ 5 puntos |
| E1 | Existen proteínas de estrategia empática (las siete estrategias de ESConv) | Aumento significativo en ≥ 5 de 7 |
| E2 | La rama emocional supera a una proteína fija y a un LoRA de empatía con el mismo presupuesto | Supera al LoRA en evaluación humana a ciegas |
| E3 | La rama inhibitoria reduce la adulación sin restar empatía | Adulación ≥ 30 % menor; empatía no inferior |
| C1 | La consolidación en dos fases mantiene la biblioteca al crecer | Olvido menor que la línea base desde 10 proteínas |
| C2 | Una proteína del modelo maestro funciona en el alumno mediante un traductor | Igual o mejor que la propia |
| R1 | La recuperación espaciada sin errores es viable en personas con deterioro cognitivo leve | Adherencia ≥ 70 %; retención superior al control activo |

Los umbrales son provisionales hasta medir las líneas base y se preregistran antes de cada ola.

## Olas

**Ola 0 · Cimientos.** Métrica sobre la respuesta completa, muestras de 200 casos y cinco semillas, extracción por cabezas con mediación causal, prueba en cuantización int4 real, preregistro de D1–D4.

**Ola 1 · Dendrita v1.** Ramas, compuerta de coincidencia, inhibición competitiva y presupuesto de norma (D1–D4). Primer preprint.

**Ola 2 · Rama emocional.** Líneas base (Gemma 4 sin modificar y LoRA de empatía), conjunto de evaluación en español, proteínas de estrategia empática, vectores de emoción y adulación en Gemma 4, rama inhibitoria (E1, E3).

**Ola 3 · Integración y consolidación.** Dendrita emocional completa con puerto apical (E2), consolidación en dos fases con repetición espaciada (C1). Segundo preprint.

**Ola 4 · Dispositivo.** Inyección en el móvil (llama.cpp con vectores de control o runtime propio), transferencia maestro → alumno (C2), medidas de velocidad, memoria y batería.

**Ola 5 · Personas.** Estudio de viabilidad con comité de ética, consentimiento y control activo (R1), con el modelo empaquetado en una aplicación que no saca datos del dispositivo.

**Ola 6 · Publicación.** Artículo de síntesis, ficha del modelo, repositorio de versiones firmadas.

## Protocolo común

Preregistro; al menos 200 casos por tarea, cinco semillas e intervalos de confianza; medida sobre la respuesta generada, no solo sobre el primer token; ablación de cada componente; evaluación humana a ciegas como criterio principal en las hipótesis emocionales; versión en español de cada prueba; código, semillas y artefactos con su hash. Los resultados negativos se publican.
