"""
ingest.py — сбор исходных данных и наполнение векторного индекса.

ПРАКТИКА 2. Что делает студент:
  1. clean_text()  — предварительная очистка: переносы, колонтитулы, мусорные пробелы
  2. chunk_text()  — нарезка на чанки фиксированного размера с перекрытием
  3. detect_category() — рубрикация чанка; словарь рубрик задаёте вы под свой домен

  4. черновик gold_dataset.json — 5 вопросов с указанием файлов-источников

Готово и трогать не нужно: обход data/raw/, чтение PDF и TXT, запись в ChromaDB,
замер hit-rate в evaluate_retrieval().

Каждый запуск пересобирает индекс с нуля — коллекция очищается перед наполнением.
Поэтому при подборе параметров достаточно поправить CHUNK_SIZE или CHUNK_OVERLAP
в config.py и запустить ingest.py снова: удалять data/chroma/ руками не нужно.

Критерии приёмки:
  - python ingest.py отрабатывает без ошибок, коллекция непуста
  - у каждого чанка заполнены source, page и category; пустых category нет
  - поиск с фильтром по category возвращает чанки только этой категории
  - в gold_dataset.json не меньше 5 вопросов, у каждого заполнено expected_sources
  - показаны не менее двух замеров с разными CHUNK_SIZE / CHUNK_OVERLAP,
    и выбор итоговых значений объяснён числом, а не вкусом
  - CHUNK_SIZE и CHUNK_OVERLAP берутся из config.py, а не зашиты в код

Про метрику. hit-rate@k отвечает на один вопрос: попал ли фрагмент с ответом
в первые k результатов. Считается механически, сверкой поля source с ожидаемым —
ни модели, ни судьи для этого не нужно, поэтому мерить можно уже здесь, на П2.
Не путайте её с Context Precision из триады RAG: та оценивает, насколько найденное
относится к делу, её считает LLM-судья, и появляется она только на П6. В плане
дисциплины обе назывались «точностью извлечения контекста», хотя это разные вещи.

Запуск:
    python ingest.py          # проиндексировать и сразу замерить
    python ingest.py --eval   # только замер, без переиндексации
"""

import hashlib
from pathlib import Path

import config
import re

# --- Читатели форматов: готовы ---

def read_txt(path: Path) -> list[tuple[int, str]]:
    """Возвращает [(номер страницы, текст)]. У TXT страниц нет, поэтому страница 0."""
    return [(0, path.read_text(encoding="utf-8", errors="ignore"))]


def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Возвращает [(номер страницы, текст)] — постранично, чтобы метаданное page было честным."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


READERS = {".txt": read_txt, ".md": read_txt, ".pdf": read_pdf}


# --- Ваша часть (П2) ---

# Словарь рубрик под ПДД РФ.
# Слова обрезаны до основы, чтобы ловить падежи. У каждой рубрики — свой набор
# сильных ключей; они выбраны так, чтобы не пересекаться с чужими рубриками.
RUBRICS = {
    "priority": [
        "перекрёстк", "перекрестк", "преимуществ", "уступить", "уступает",
        "уступи дорогу", "главн", "второстепенн", "равнозначн", "помех",
        "приоритет", "очередност", "13.9", "13.10", "13.11", "13.12", "13.13",
        "13.4", "13.5", "13.6", "13.7", "13.8",
    ],
    "signs": [
        "знак", "табличк", "разметк", "светофор", "регулировщик",
        "предупреждающ", "запрещающ", "предписывающ", "знаки приоритета",
        "знаки сервиса", "знак 6.4", "знак 8.17", "знак 5.19", "знак 1.14",
        "знак 3.24", "5.19.1", "5.19.2", "6.4 ", "8.17 ", "1.14.1", "1.14.2",
        "приложение 1", "приложение 2",
    ],
    "movement": [
        "маневрирован", "перестроени", "поворот", "разворот", "обгон",
        "опережени", "встречн", "полос", "ряд", "трамвайны пут",
        "скорост", "10.1", "10.2", "10.3", "10.4", "10.5",
        "остановк", "стоянк", "парковк", "12.1", "12.2", "12.4", "12.5",
        "ближн свет", "дальн свет", "габаритн", "противотуманн",
        "звуков сигнал", "19.1", "19.2", "19.5",
        "начало движени", "8.1", "8.2", "8.3", "18.1", "18.2",
        "задним ходом", "буксировк",
    ],
    "duties": [
        "обязанност", "обязан ", "водител механическ", "документ",
        "водительск удостовер", "регистрационн документ",
        "ремн безопасност", "мотошлем", "пристегнут",
        "учебн езд", "обучающ вождени", "обучающ", "автодром",
        "перевозк людей", "перевозк груз", "перевозк детей",
        "детск удерживающ", "организованн перевозк",
        "время управлени", "отдых водител", "тахограф", "26.1", "26.2", "26.3",
        "основные поняти", "основные термин", "1.2.",
    ],
    "responsibility": [
    "штраф", "коап", "ответственност", "наказани", "лишени",
    "нарушени", "влечет", "наказывается",
    "12.8", "12.9", "12.12", "12.13", "12.15", "12.16", "12.17",
    "12.18", "12.19", "12.21", "12.24", "12.26", "12.27", "12.37",
    "административн", "правонарушени",
    ]
    
}
DEFAULT_RUBRIC = "general"


def clean_text(text: str) -> str:
    r"""Очистить сырой текст перед нарезкой.

    Специфика этого корпуса (pddmaster.ru, PDF с картинками знаков):
      - колонтитул "Актуальные правила дорожного движения..." на каждой странице;
      - рекламные ссылки в конце файла;
      - титульный лист и оглавление с точечными заполнителями ("...... 5");
      - слова, разорванные переносом: "уста-\nнавливают" -> "устанавливают";
      - цепочки пробелов и пустых строк от вёрстки PDF.

    Про re.DOTALL: с жадным .* он съедает весь файл до конца, а не одну строку.
    Поэтому колонтитулы и рекламу убираем построчно через re.MULTILINE и ^...$.
    """
    if not text:
        return ""

    # 1) Колонтитул pddmaster.ru — повторяется на каждой странице.
    text = re.sub(
        r"^[ \t]*Актуальные правила дорожного движения.*$",
        "", text, flags=re.MULTILINE,
    )
    text = re.sub(r"https?://pddmaster\.ru\S*", "", text)
    text = re.sub(r"https?://pdd-mail\.ru\S*", "", text)

    # 2) Рекламный блок в конце файла.
    text = re.sub(r"^[ \t]*Правила дорожного движения online:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*Мобильное приложение.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*Экзаменационный билеты ГИБДД online:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*Таблица штрафов ГИБДД:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*Договор купли-продажи автомобиля:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*Договор дарения автомобиля:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*Калькулятор стоимости ОСАГО:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*Таблица с кодами регионов.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*И еще множество полезной информации.*$", "", text, flags=re.MULTILINE)

    # 3) Титульный лист и выходные данные.
    text = re.sub(r"Правила Дорожного Движения \d{4}-\d{4} от pddmaster\.ru", "", text)
    text = re.sub(r"Редакция от .*?Версия\s*[–-]\s*[\d.]+", "", text)
    text = re.sub(r"Оглавление\s+Оглавление", "", text)

    # 4) Оглавление: точечные заполнители "....... 5", "............ 11".
    #    Сами заголовки разделов остаются — они могут быть полезны поиску.
    text = re.sub(r"\.{4,}\s*\d*", " ", text)

    # 5) Склейка слов, разорванных переносом: "уста-\nнавл" -> "устанавливают"
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"-\s+", "", text)

    # 6) Сжатие пробелов и пустых строк
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Нарезать текст на чанки размером size с перекрытием overlap.

    Шаг сдвига = size - overlap. Перекрытие — хвост предыдущего чанка,
    который повторяется в начале следующего, чтобы мысль на границе
    нарезки не потерялась.
    """
    if size <= 0:
        raise ValueError("size должен быть > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap должен быть в [0, size)")

    chunks: list[str] = []
    step = size - overlap
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def detect_category(text: str, source: str) -> str:
    """Определить рубрику чанка по ключевым словам.

    Рубрика нужна для фильтрации при поиске (where={'category': ...}).
    Всегда что-то возвращает: неопознанное падает в DEFAULT_RUBRIC,
    чтобы фильтрованный поиск не терял фрагменты.
    """
    t = text.lower()
    for rubric, keywords in RUBRICS.items():
        for kw in keywords:
            if kw in t:
                return rubric
    return DEFAULT_RUBRIC

# --- Сборка индекса: готова ---

def ingest() -> int:
    """Обходит исходные данные, нарезает и складывает в ChromaDB. Возвращает число чанков.

    Индекс собирается заново с нуля: коллекция очищается перед наполнением.
    Иначе после смены CHUNK_SIZE в коллекции остались бы чанки предыдущей
    нарезки, и hit-rate@5 замерялся бы по смеси двух — молча, без ошибки.
    """
    collection = config.reset_collection()
    total = 0

    # README.md отсеивается намеренно: это шаблонная заглушка каталога,
    # а не документ предметной области. Без отсева она попадает в индекс
    # и выдаётся поиском как знание.
    files = [p for p in config.RAW_DIR.rglob("*")
             if p.suffix.lower() in READERS and p.name != "README.md"]
    if not files:
        raise SystemExit(f"В {config.RAW_DIR} нет файлов .txt/.md/.pdf — положите исходные данные (П2).")

    for path in files:
        for page, raw in READERS[path.suffix.lower()](path):
            cleaned = clean_text(raw)
            if not cleaned:
                continue

            for chunk in chunk_text(cleaned):
                # id детерминированный: одинаковый текст даёт один и тот же чанк,
                # поэтому дубли не возникают даже внутри одного прогона
                chunk_id = hashlib.sha1(
                    f"{path.name}:{page}:{chunk}".encode("utf-8")
                ).hexdigest()

                collection.upsert(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": path.name,
                        "page": page,
                        "category": detect_category(chunk, path.name),
                    }],
                )
                total += 1

    return total


# --- Замер качества поиска: готов ---

def evaluate_retrieval(k: int = config.TOP_K) -> float:
    """hit-rate@k по черновику Gold Dataset. Готово, менять не нужно.

    Для каждого вопроса выполняется поиск и проверяется, попал ли в первые k
    результатов хотя бы один чанк из ожидаемых источников.

    Вопросы с пустым expected_sources пропускаются: это проверка отказа отвечать,
    а она требует LLM и меряется на П6, а не здесь.

    Обращается к коллекции напрямую, а не через MCP: на П2 оркестратора ещё нет.
    """
    import json

    dataset_path = Path(__file__).parent / "gold_dataset.json"
    cases = [c for c in json.loads(dataset_path.read_text(encoding="utf-8"))
             if c.get("expected_sources")]

    if not cases:
        print("В gold_dataset.json нет вопросов с expected_sources — замерять нечего (П2).")
        return 0.0

    collection = config.get_collection()
    hits = 0

    print(f"\nhit-rate@{k}  |  чанк {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}")
    for case in cases:
        result = collection.query(query_texts=[case["question"]], n_results=k)
        found = {(m or {}).get("source", "") for m in (result.get("metadatas") or [[]])[0]}
        hit = bool(found & set(case["expected_sources"]))
        hits += hit
        print(f"  [{'+' if hit else '-'}] {case['question'][:70]}")
        if not hit:
            print(f"      ожидалось: {case['expected_sources']}, найдено: {sorted(found)}")

    rate = hits / len(cases)
    print(f"\nhit-rate@{k} = {rate:.0%}  ({hits} из {len(cases)})")
    skipped = len(json.loads(dataset_path.read_text(encoding="utf-8"))) - len(cases)
    if skipped:
        print(f"Пропущено вопросов без expected_sources: {skipped} — они меряются на П6.")
    return rate


if __name__ == "__main__":
    import sys

    if "--eval" not in sys.argv:
        count = ingest()
        print(f"Индекс пересобран заново. Проиндексировано чанков: {count}")
        print(f"Всего в коллекции: {config.get_collection().count()}")

    evaluate_retrieval()
