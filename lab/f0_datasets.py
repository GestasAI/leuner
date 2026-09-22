"""Datasets de juguete para el Experimento F0 (viabilidad de la proteína).

Tareas clásicas, verificables y de respuesta corta (idealmente 1 token), para medir
si un vector-tarea inyectado hace que el modelo resuelva la tarea en frío.

Cada tarea es una lista de pares (entrada, salida). El runner separa pool de
demostraciones y conjunto de test con una semilla fija.
"""

# Antónimos en inglés (respuestas de 1 token frecuente: bueno para first-token match).
ANTONYM = [
    ("good", "bad"), ("hot", "cold"), ("up", "down"), ("big", "small"),
    ("fast", "slow"), ("happy", "sad"), ("light", "dark"), ("high", "low"),
    ("open", "closed"), ("rich", "poor"), ("strong", "weak"), ("hard", "soft"),
    ("full", "empty"), ("clean", "dirty"), ("early", "late"), ("near", "far"),
    ("true", "false"), ("win", "lose"), ("buy", "sell"), ("love", "hate"),
    ("day", "night"), ("left", "right"), ("old", "young"), ("wet", "dry"),
    ("loud", "quiet"), ("sharp", "dull"), ("thick", "thin"), ("wide", "narrow"),
    ("deep", "shallow"), ("heavy", "light"), ("smooth", "rough"), ("tight", "loose"),
    ("brave", "afraid"), ("first", "last"), ("more", "less"), ("yes", "no"),
    ("begin", "end"), ("accept", "reject"), ("expand", "shrink"), ("arrive", "depart"),
]

# País -> capital (muchas capitales son 1 token en el tokenizer de Gemma).
CAPITAL = [
    ("France", "Paris"), ("Spain", "Madrid"), ("Italy", "Rome"), ("Germany", "Berlin"),
    ("Portugal", "Lisbon"), ("Greece", "Athens"), ("Japan", "Tokyo"), ("China", "Beijing"),
    ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Cuba", "Havana"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Norway", "Oslo"),
    ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Ireland", "Dublin"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Turkey", "Ankara"), ("India", "Delhi"), ("Thailand", "Bangkok"),
    ("Kenya", "Nairobi"), ("Mexico", "Mexico"), ("Canada", "Ottawa"), ("Brazil", "Brasilia"),
    ("Argentina", "Buenos"), ("Colombia", "Bogota"), ("Hungary", "Budapest"), ("Denmark", "Copenhagen"),
    ("Belgium", "Brussels"), ("Morocco", "Rabat"), ("Lebanon", "Beirut"), ("Jordan", "Amman"),
    ("Vietnam", "Hanoi"), ("Ukraine", "Kyiv"), ("Croatia", "Zagreb"), ("Serbia", "Belgrade"),
]

TASKS = {
    "antonym": ANTONYM,
    "capital": CAPITAL,
}

# Plantillas de prompt por tarea.
# Las entradas se insertan como: PREFIX + x + SEP + " " + y  (demos)
#                                PREFIX + x + SEP              (query)
# Usar ": " como SEP (funciona mejor que "->" en modelos IT fuertemente ajustados).
TASK_TEMPLATES = {
    "antonym": {"prefix": "opposite of ", "sep": ":"},
    "capital": {"prefix": "Capital of ", "sep": ":"},
}
