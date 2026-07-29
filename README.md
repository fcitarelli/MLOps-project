# MLOps-project

Progetto finale del corso MLOps (ProfessionAI). Pipeline end-to-end per un
modello di sentiment analysis su testi da social media: fine-tuning del
modello, pipeline CI/CD (training, test, deploy simulato) e monitoraggio
continuo delle performance.

## Indice

- [Fase 1 — Modello di sentiment analysis](#fase-1--modello-di-sentiment-analysis)
- [Fase 2 — Pipeline CI/CD](#fase-2--pipeline-cicd)
- [Fase 3 — Deploy e monitoraggio](#fase-3--deploy-e-monitoraggio)
- [Struttura del progetto](#struttura-del-progetto)
- [Come eseguire il progetto in locale](#come-eseguire-il-progetto-in-locale)
- [Limiti noti ed estensioni possibili](#limiti-noti-ed-estensioni-possibili)

## Fase 1 — Modello di sentiment analysis

### Scelte progettuali

- **Modello base**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
  RoBERTa già pre-addestrato per sentiment su testi Twitter. Scelto perché
  già allineato al dominio (social media) richiesto dalla consegna, invece
  di partire da un RoBERTa generico.
- **Dataset**: [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
  (config `sentiment`), 3 classi: `negative` / `neutral` / `positive`.
  Split nativi `train` (45.615 righe), `validation` (2.000), `test`
  (12.284) — stesso dataset con cui il modello base è stato originariamente
  valutato, quindi coerente con il task.
- **Preprocessing** (`src/preprocessing.py`): normalizzazione stile
  CardiffNLP — ogni `@menzione` diventa `@user`, ogni URL diventa `http`.
  Riduce la sparsità del vocabolario dovuta a username/link unici per
  ogni tweet, mantenendo lo stesso preprocessing usato per pre-addestrare
  il modello base.
- **Limiti sui campioni** (`src/config.py`: `MAX_TRAIN_SAMPLES=1000`,
  `MAX_VAL_SAMPLES=250`): il training gira su CPU (locale e in CI, nessuna
  GPU disponibile), quindi il dataset viene sotto-campionato per tenere i
  tempi ragionevoli. Compromesso esplicito: risultati da corso, non da
  produzione.

### Implementazione

- `src/dataset.py`: carica il dataset, applica preprocessing e
  tokenizzazione (`max_length=128`, padding/truncation), rinomina `label`
  in `labels` (richiesto da `Trainer`), applica i limiti di campionamento.
- `src/training.py`: fine-tuning con `transformers.Trainer`
  (`learning_rate=2e-5`, `batch_size=8`, `epochs=2`), valutazione a ogni
  epoca, `load_best_model_at_end=True` selezionato su `f1` (macro),
  salvataggio del modello migliore in `models/sentiment_model/`.
- `src/evaluation.py`: `compute_metrics` — accuracy, precision/recall/f1
  macro (macro perché le classi non sono perfettamente bilanciate).
- `src/inference.py`: carica il modello salvato ed espone
  `predict(text, log=True)` → `{label, confidence, scores}`. Se `log=True`
  registra la predizione in `monitoring/logs/predictions.jsonl` (usato in
  Fase 3).

### Risultati ottenuti

Metriche di validazione registrate durante il training (`results/checkpoint-*/trainer_state.json`):

| Epoca | Accuracy | F1 (macro) | Precision (macro) | Recall (macro) |
|-------|----------|------------|--------------------|-----------------|
| 1     | 0.728    | 0.722      | 0.736              | 0.716           |
| 2     | 0.740    | 0.733      | 0.739              | 0.728           |

Il checkpoint dell'epoca 2 ha F1 più alto ed è quello effettivamente
salvato in `models/sentiment_model/` (selezione automatica di
`load_best_model_at_end`). Con solo 1.000 esempi di training i numeri sono
modesti ma coerenti con un training rapido su CPU; con il dataset completo
ci si aspetta un miglioramento sensibile.

## Fase 2 — Pipeline CI/CD

### Scelte progettuali

- **GitHub Actions** (`.github/workflows/ci-cd.yml`): scelto perché nativo
  del repository, nessuna infrastruttura da gestire.
- **Tre job in sequenza, non un unico script**: `train` → `test` → `deploy`
  (più `monitor` in parallelo dopo `train`). Separarli rende esplicito
  dove fallisce la pipeline (training rotto vs test rotto vs deploy
  rotto) e permette di scaricare/ispezionare ogni artifact singolarmente.
- **Passaggio artifact tra job** (`actions/upload-artifact` /
  `download-artifact`): ogni job gira su un runner pulito, quindi il
  modello addestrato in `train` viene esplicitamente caricato e poi
  riscaricato nei job successivi — nessuno stato condiviso implicito.
- **Trigger**: `push` e `pull_request` su `main`. Ogni modifica al
  codice fa ripartire l'intera pipeline (retraining incluso), coerente
  con l'idea di CI/CD "il main è sempre verificato end-to-end".

### Implementazione

| Job       | Cosa fa |
|-----------|---------|
| `train`   | Installa `requirements.txt`, esegue `train.py` da zero, carica `models/sentiment_model/` come artifact `sentiment-model`. |
| `test`    | Scarica `sentiment-model`, esegue `pytest tests/ -v`. |
| `deploy`  | Scarica `sentiment-model`, esegue `python -m deploy.deploy`, carica il pacchetto risultante come artifact `huggingface-space-package`. |
| `monitor` | Scarica `sentiment-model`, esegue `monitoring/build_dataset.py` + `monitoring/report.py`, carica il report come artifact `monitoring-report`. |

Test (`tests/`):

- `test_preprocessing.py` — unit test su normalizzazione `@user`/`http` e
  su forma dell'output di tokenizzazione.
- `test_evaluation.py` — unit test su `compute_metrics` con predizioni
  sintetiche note.
- `test_inference.py` — integration test: carica il modello reale
  addestrato e verifica che `predict()` ritorni una label valida, una
  confidence in `[0, 1]` e scores che sommano a 1. Marcato `skipif` se il
  modello non è presente (ambiente locale senza training già eseguito).

Deploy simulato (`deploy/`):

- `deploy/app.py` — piccola app Gradio che avvolge `predict()`.
- `deploy/README_space.md` — README con frontmatter in stile Hugging Face
  Space (`sdk: gradio`, ecc.).
- `deploy/deploy.py` — assembla `dist/huggingface_space/` (app + modello +
  `requirements.txt` + README) e stampa il comando che *verrebbe*
  eseguito (`huggingface-cli upload ...`), senza chiamare davvero
  l'API di Hugging Face e senza richiedere un token.

## Fase 3 — Deploy e monitoraggio

### Scelte progettuali

- **Evidently** invece di uno stack Prometheus/Grafana: libreria Python,
  nessun servizio da tenere sempre attivo, genera report HTML/JSON
  autonomi — proporzionato a un progetto senza traffico reale continuo.
- **Niente traffico reale disponibile** → il monitoraggio confronta due
  dataset etichettati invece di dati di produzione veri:
  - **reference** = split `validation` (la stessa distribuzione usata per
    scegliere il checkpoint migliore in fase di training);
  - **current** = split `test`, usato come proxy di "nuovi dati osservati
    in produzione".

  Questo permette di calcolare sia data drift sia un vero calo di
  accuracy (avendo il target reale su entrambi i lati), invece di un
  monitoraggio solo sulla distribuzione degli input.
- **Log delle predizioni singole** (`monitoring/logs/predictions.jsonl`,
  scritto da `predict(..., log=True)`): rappresenta il traffico reale
  che arriverebbe dall'app Gradio in produzione: sentiment rilevato,
  confidence, timestamp.

### Implementazione

- `monitoring/build_dataset.py`: per ogni riga di `validation`/`test`
  (limitate a `MAX_REFERENCE_SAMPLES`/`MAX_CURRENT_SAMPLES=200` per
  velocità) esegue `predict()` e salva `text`, `text_length`, `target`
  (label vera), `prediction`, `confidence` in
  `monitoring/data/reference.csv` e `current.csv`.
- `monitoring/report.py`: costruisce due `evidently.Dataset` con
  `DataDefinition` (colonne numeriche `confidence`/`text_length`,
  classificazione multiclasse `target`/`prediction`), esegue un
  `Report` con `ClassificationPreset` (accuracy, F1, confusion matrix)
  + `DataDriftPreset` (drift sulle feature numeriche), salva
  `monitoring/reports/report.html` e `.json`. Stampa inoltre un confronto
  diretto di accuracy reference vs current, con warning se il calo
  supera la soglia `ACCURACY_DROP_THRESHOLD = 0.1`.

### Risultati ottenuti

Da un'esecuzione locale di esempio:

```
[monitoring] reference accuracy: 0.755
[monitoring] current accuracy: 0.710
[monitoring] accuracy within threshold (drop=0.045)
```

Calo di accuracy sotto soglia (0.045 < 0.1) tra `validation` e `test`:
nessun alert. Il report HTML completo (drift per colonna, confusion
matrix, distribuzioni) è consultabile aprendo
`monitoring/reports/report.html` dopo l'esecuzione.

## Struttura del progetto

```
.
├── train.py                     # entry point training
├── src/
│   ├── config.py                # iperparametri, path, costanti monitoring
│   ├── dataset.py                # caricamento + preprocessing dataset
│   ├── preprocessing.py          # normalizzazione testo + tokenizer
│   ├── training.py                # training loop (Trainer)
│   ├── evaluation.py              # metriche di valutazione
│   ├── inference.py               # predict() + logging predizioni
│   └── monitoring.py               # logger jsonl delle predizioni
├── tests/                        # unit + integration test (pytest)
├── deploy/
│   ├── app.py                    # app Gradio
│   ├── README_space.md            # README stile HF Space
│   └── deploy.py                   # assembla pacchetto + deploy simulato
├── monitoring/
│   ├── build_dataset.py           # genera reference.csv / current.csv
│   └── report.py                   # report Evidently + check soglia
├── .github/workflows/ci-cd.yml   # pipeline CI/CD (train/test/deploy/monitor)
├── models/                       # modello addestrato (generato, non versionato)
├── results/                      # checkpoint di training (generato, non versionato)
└── requirements.txt
```

## Come eseguire il progetto in locale

```bash
pip install -r requirements.txt

# training (genera models/sentiment_model)
python train.py

# test
pytest tests/ -v

# deploy simulato (genera dist/huggingface_space)
python -m deploy.deploy

# monitoring (genera monitoring/data e monitoring/reports/report.html)
python -m monitoring.build_dataset
python -m monitoring.report
```

## Limiti noti ed estensioni possibili

- Training limitato a 1.000/250 campioni per velocità su CPU: con
  dataset completo e/o GPU le metriche migliorerebbero.
- Deploy e monitoraggio sono simulati (nessun token, nessuna chiamata
  esterna reale): per un deploy reale basterebbe sostituire il print in
  `deploy/deploy.py` con una chiamata a `huggingface_hub.upload_folder`
  autenticata via token.
- Il monitoraggio confronta due split statici del dataset invece di
  traffico di produzione reale, in assenza di un'app effettivamente
  esposta con utenti: `monitoring/logs/predictions.jsonl` è già pronto a
  raccogliere richieste reali se l'app venisse esposta pubblicamente.
