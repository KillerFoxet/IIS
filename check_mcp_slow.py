"""Проверка MCP-поиска с длинным таймаутом.

Первый вызов MCPTools занимает больше стандартных 120 секунд:
сервер запускает подпроцесс, поднимает ChromaDB и загружает модель
эмбеддингов multilingual-e5-small (~470 МБ). Дальше вызовы быстрые.
"""

from mcp_client import MCPTools

tools = MCPTools(timeout=600.0)   # 10 минут на первый вызов
try:
    print("Сервер поднимается, первый запрос может занять 2–5 минут...")
    chunks = tools.search("Кто имеет преимущество на перекрёстке равнозначных дорог?")
    print(f"\nНайдено чанков: {len(chunks)}\n")
    for c in chunks:
        page = f" стр. {c.page}" if c.page else ""
        print(f"{c.score:.3f} {c.source}{page} [{c.category}]")
finally:
    tools.close()