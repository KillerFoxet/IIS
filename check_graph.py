"""Проверка графа переходов без запуска оркестратора:
достижимость и сироты."""
from orchestrator import Orchestrator as O
roles = set(O.ROLES)
for role in O.ROLES:
    allowed = O.TRANSITIONS.get(role, set())
    print(f"{role:<11} -> {sorted(allowed)} запрещено: {sorted(roles - allowed)}")
seen, todo = set(), [O.INITIAL_ROLE]
while todo:
    role = todo.pop()
    if role not in seen:
        seen.add(role)
        todo += O.TRANSITIONS.get(role, set())
print("недостижимы:", sorted(roles - seen) or "нет")
print("без интента:", sorted(roles -
set(O.INTENT_TO_ROLE.values())) or "нет")
named = {O.INITIAL_ROLE, O.FALLBACK_ROLE,
*O.INTENT_TO_ROLE.values(), *O.TRANSITIONS}
print("названы, но не описаны в ROLES:", sorted(named - roles)
or "нет")