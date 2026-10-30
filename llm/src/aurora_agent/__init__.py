"""Aurora agent core: the language model proposes, plain code decides.

Layers, from the client inward:
  planner.py   language model (or keyword fallback) turns a message into proposed actions
  gateway.py   deterministic rule check: allow / ask the client / second key / deny
  rules.py     the client's own rules; stricter changes apply at once, looser ones after 24 hours
  ledger.py    append-only, hash-chained journal of every proposal and decision
  agent.py     orchestration: confirmations, the second key, execution
"""
