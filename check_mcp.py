from orchestrator import Orchestrator

o = Orchestrator()
try:
    chunks = o.tools.search("Кто имеет преимущество на перекрёстке равнозначных дорог?")
    for c in chunks:
        page = f" стр. {c.page}" if c.page else ""
        print(f"{c.score:.3f} {c.source}{page} [{c.category}]")
finally:
    o.close()