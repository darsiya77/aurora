"""Aurora assistant: a chat model that answers any question and performs actions through tools.

  tools.py     what the model may call, and how each call reaches the agent core
  format.py    one text format for tool calls that works with any chat model
  data.py      training dialogs: actions, answers, refusals
  train.py     LoRA fine-tuning loop (plain PyTorch, every step visible)
  evaluate.py  how often the model calls the right tool with the right arguments
  assistant.py the runtime loop: model -> tool call -> rule check -> answer
"""
