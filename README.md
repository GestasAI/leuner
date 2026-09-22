# Launer

**Una dendrita artificial dentro de un modelo de lenguaje abierto.**
Inteligencia emocional para acompañar, construida con partes diminutas sobre Gemma 4.

Launer es el programa de investigación de [GestasAI](https://gestasai.com/). Toma la dendrita, la parte de la neurona que recibe y decide, como plano para construir una unidad de inteligencia emocional dentro de un modelo de lenguaje abierto. No proponemos un modelo más grande, sino uno hecho de partes pequeñas y especializadas, organizadas como la naturaleza organiza una dendrita: recibe una señal, decide en local y en paralelo, se mira en el resultado, deja que un moderador frene o deje pasar, y consolida mientras descansa.

Ensayo científico completo: **[launer.es](https://launer.es)** · Informe LNR-2026-01, versión 0.2.

> *Launer takes the dendrite as the blueprint for an emotional-intelligence unit built inside an open language model (Gemma 4). First measured milestone: a 6 KB task vector injected at layer 20/35 of Gemma 4 E2B drives zero-shot translation from 0 % to 90 %. This repository holds the laboratory code, the extracted vectors, the measurement reports and the source of the scientific website.*

## Regla del proyecto

Solo lo medido se comunica como resultado. Todo lo demás se etiqueta como **diseñado** o **hipótesis**. Los resultados negativos también se publican.

## Lo que está medido

| Resultado | Cifra | Dónde |
|---|---|---|
| Una proteína (vector de 1.536 dimensiones, 6 KB) inyectada en la capa 20 de Gemma 4 E2B activa la traducción sin ejemplos en el contexto | 0 % → 90 % (media 80 % ± 9 en cinco semillas) | `lab/`, F0–F4 |
| Lo mismo para capitales y para traducción español–inglés | 0 % → 90 % y 0 % → 93 % | F0, F5 |
| Una proteína fuera de su tarea es inerte o impone su capacidad: no es multitarea | 0 % en seis tareas cruzadas | F8 |
| Una proteína extraída de experiencia mezclada (capitales + traducción) activa las dos capacidades | 93 % y 53 % | F8 |
| Tres proteínas encadenadas por un enrutador de decisión tipada resuelven una pregunta de tres pasos (Mozart → Austria → Viena) | 80 % de extremo a extremo; base sin proteínas 0 % | F9 |

Muestras de 15 a 20 casos y una semilla salvo donde se indica. Son los primeros números, no resultados cerrados; la hoja de ruta fija cómo se consolidan.

## La dendrita, en cinco fases

| Fase | Qué hace | En la naturaleza | Estado |
|---|---|---|---|
| 1 · Proteína | Recibe: un vector extraído del modelo activa una capacidad | Neurotransmisor y espina | **Medida** |
| 2 · Decisión tipada | Decide en local y en paralelo, con probabilidad calibrada | Espiga dendrítica | Borrador medido (F9) |
| 3 · Conducto espejo | Se mira en el resultado: estado de pocas dimensiones | Potencial de acción retropropagado | Diseño |
| 4 · Moderador | Frena, deja pasar o escala; separa proteínas | Interneuronas y neuromoduladores | Diseño |
| 5 · Consolidación | Repasa, poda y espacia fuera de línea | Sueño NREM y REM | Diseño |

Detalle, estudios y correspondencias en el [ensayo](https://launer.es). Plan en [ROADMAP.md](ROADMAP.md).

## Contenido del repositorio

```
lab/        laboratorio de proteínas: extracción, inyección, medición y orquestación (Python)
lab/proteins/   vectores extraídos (.f32 + .json con hash SHA-256)
lab/reports/    informes de medición en JSON
web/        fuente del ensayo científico de launer.es y su constructor
```

Guía del laboratorio, con el método y todos los resultados: [lab/README.md](lab/README.md).

## Reproducir el hito

Hace falta Python 3.10+, un PC con 16 GB de RAM (funciona en CPU) y aceptar la licencia de Gemma en Hugging Face.

```bash
cd lab
pip install -r requirements.txt
huggingface-cli login
python f0_function_vector.py --model google/gemma-4-E2B-it --task capital --layers 20
```

El script imprime el acierto sin vector, con ejemplos en el contexto y con el vector inyectado. Con la semilla por defecto reproduce el 0 % → 90 % del informe.

## Modelo base

[Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4) de Google DeepMind, licencia Apache 2.0. Launer no reentrena el modelo: le añade dendritas. Los pesos no están en este repositorio.

## Cómo citar

```bibtex
@techreport{launer2026dendrita,
  title       = {Construir una dendrita dentro de un modelo de lenguaje:
                 Launer, una inteligencia emocional artificial hecha de partes diminutas},
  author      = {Aguirre P{\'e}rez, Juan Carlos},
  institution = {GestasAI},
  number      = {LNR-2026-01},
  year        = {2026},
  url         = {https://launer.es/}
}
```

También en [CITATION.cff](CITATION.cff).

## Licencia y contacto

Código, vectores y textos de este repositorio: [Apache 2.0](LICENSE).
Autor: [Juan Carlos Aguirre Pérez](https://juancarlosaguirre.es), GestasAI · info@gestasai.com
