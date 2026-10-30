# Aurora AI

Два пакета:

- **`aurora_agent`** — ядро ИИ-агента Авроры: модель предлагает, обычный код проверяет правила клиента и решает. Архитектура, модель угроз и точки интеграции с банком — в [`docs/AGENT.md`](docs/AGENT.md).
- **`aurora_assistant`** — ассистент, который отвечает на любые вопросы и выполняет действия: открытая модель семейства Kimi, дообученная вызывать инструменты Авроры, и веб-чат, где её можно учить. Подробно — в [`docs/ASSISTANT.md`](docs/ASSISTANT.md).
- **`aurora_llm`** — собственная языковая модель: от сырого текста до HTTP-сервера, к которому подключается агент. Всё написано с нуля на PyTorch, каждую часть можно прочитать и поменять.

## Агент за минуту

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m aurora_agent.demo
```

Демо проводит через семь ситуаций: план на зарплату, покупку сверх лимита, панику, звонок «службы безопасности», подложенную команду в новости, крупный вывод с подтверждением близкого, изменение правил. В конце подделывает запись журнала и показывает, что подмена найдена.

## Своя модель: с чего начать

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_tiny.sh          # весь путь на процессоре за ~15 минут
```

Скрипт скачает 3000 статей русской Википедии, обучит токенизатор, подготовит данные, обучит маленькую модель, покажет её текст и дообучит на примерах диалогов.

## Что внутри

```
configs/                 размеры моделей: tiny, small, base, xl
src/aurora_agent/        ядро агента: планировщик, проверка правил, правила, журнал, демо
src/aurora_assistant/    ассистент: инструменты, данные, дообучение LoRA, оценка, веб-чат
src/aurora_llm/
  model.py               сама модель (трансформер-декодер)
  tokenizer.py           обучение и работа токенизатора
  data/                  загрузка, чистка, дедупликация, нарезка на токены
  train.py               предобучение, в том числе на нескольких GPU
  sft/train_sft.py       дообучение на диалогах
  chat.py                формат диалога
  generate.py            генерация текста и чат в терминале
  eval/perplexity.py     оценка качества
  serve.py               HTTP-сервер в формате, совместимом с OpenAI
scripts/                 загрузка данных, пример диалогов, полный прогон
tests/                   автоматические проверки
docs/
  AGENT.md               архитектура агента, модель угроз, интеграция с банком
  ASSISTANT.md           ассистент: модель, дообучение, чат, как учить через чат
  AURORA_CONTEXT.md      что ассистент знает о проекте
  ROADMAP.md             план по этапам
  COMPUTE.md             сколько стоит обучение и какое нужно железо
  DATA.md                откуда брать данные и как их чистить
  GLOSSARY.md            словарь терминов простыми словами
```

## Команды по шагам

```bash
export PYTHONPATH=src
python scripts/download_sample.py --docs 3000                       # данные
python -m aurora_llm.tokenizer --raw data/raw --vocab 8000          # токенизатор
python -m aurora_llm.data.prepare --raw data/raw --out data/processed
python -m aurora_llm.train --config configs/tiny.yaml               # предобучение (--resume продолжает)
python -m aurora_llm.eval.perplexity --ckpt runs/tiny/best.pt       # оценка
python -m aurora_llm.generate --ckpt runs/tiny/best.pt --prompt "Москва —"
python scripts/make_sft_sample.py
python -m aurora_llm.sft.train_sft --config configs/tiny.yaml       # дообучение в ассистента
python -m aurora_llm.generate --ckpt runs/tiny-sft/best.pt --chat
python -m aurora_llm.serve --ckpt runs/tiny-sft/best.pt --port 8000
```

На нескольких видеокартах: `torchrun --nproc_per_node=8 -m aurora_llm.train --config configs/base.yaml`.

Тесты: `pip install pytest && pytest`.

## Размеры моделей

| Конфиг | Параметры | Железо | Что умеет |
|---|---|---|---|
| `tiny` | 5 млн | процессор | проверка конвейера, похожие на слова сочетания |
| `small` | 39 млн | 1 игровая видеокарта, ~3 ч | грамотные предложения |
| `base` | 330 млн | 8 GPU, ~5 ч | связные абзацы, простые факты |
| `xl` | 1,2 млрд | 8 GPU, ~2,5 дня | основа для простого ассистента |

Подробный расчёт — в `docs/COMPUTE.md`. План работы — в `docs/ROADMAP.md`.

## Честно о границах

Модель уровня лучших открытых требует сотен тысяч GPU-часов. Поэтому в плане две дорожки: растим свою модель и параллельно дообучаем готовую открытую модель для продукта. Код данных, диалогов, оценки и сервера общий для обеих.
