"""check_fsm.py — проверка автомата оркестратора на наборе фраз.

Прогоняет detect_intent() и route() на фразах под три роли (analyst, verifier,
clarifier) плюс одна заведомо непонятная. Смотрит два свойства автомата:
  1) смена интента переключает роль;
  2) непонятная фраза уходит в FALLBACK_ROLE, а не роняет процесс.
"""

import logging

logging.basicConfig(level=logging.INFO)

from orchestrator import Orchestrator

o = Orchestrator()
phrases = [
    # question -> analyst
    "Кто имеет преимущество на нерегулируемом перекрёстке равнозначных дорог?",
    "С какой максимальной скоростью можно ехать в жилой зоне?",
    # check -> verifier
    "Ты уверен, что на перекрёстке преимущество у помехи справа?",
    "Перепроверь, точно ли штраф за красный свет — тысяча рублей?",
    # situation -> clarifier
    "Можно ли развернуться?",
    "Как быть, если я не заметил знак?",
    # нераспознанное — уходит в question -> analyst
    "asdf qwerty",
]
for p in phrases:
    intent = o.detect_intent(p)
    role = o.route("test", intent)
    print(f"{p[:40]:<42} интент {intent:<12} роль {role}")
o.close()