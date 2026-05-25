# Grossistprognoser

AI-baserat prognosverktyg för grossister. Ladda upp din försäljningshistorik och få automatiska 7-dagars beställningsrekommendationer med Prophet-baserad AI.

## Funktioner

- **CSV/Excel-uppladdning** med kolumnerna `vara`, `datum`, `antal`
- **AI-prognos** 7 dagar framåt per produkt (Prophet + säsongsjustering)
- **10% säkerhetsmarginal** inbyggd i beställningsrekommendationerna
- **Statusar**: Brådskande (< 3 dagar lager) / Snart (< 7 dagar) / Planera (> 7 dagar)
- **Demo-data** — generera 90 dagars realistisk testdata med ett klick

---

## Installation

### Krav

- Python 3.10 eller senare
- pip

### 1. Installera beroenden

```bash
cd backend
pip install -r requirements.txt
```

> **OBS:** Prophet kan ta 5–10 minuter att installera första gången eftersom den
> kompilerar Stan-modeller. Ha tålamod.
>
> **Windows-tips:** Om Prophet-installationen misslyckas, prova:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

### 2. Starta backend

**Alternativ A — dubbelklicka på:**
```
start_backend.bat
```

**Alternativ B — terminal:**
```bash
cd backend
python app.py
```

Backend körs nu på `http://localhost:5000`

### 3. Öppna frontend

Öppna `frontend/index.html` direkt i webbläsaren (dubbelklicka på filen).

---

## Inloggning

| Fält     | Värde            |
|----------|------------------|
| Email    | demo@grossist.se |
| Lösenord | demo123          |

---

## CSV-format

Filen måste ha följande kolumner (svenska eller engelska namn accepteras):

| Kolumn | Alternativa namn              | Exempel        |
|--------|-------------------------------|----------------|
| vara   | produkt, product, artikel     | Mjölk 3L       |
| datum  | date, dag                     | 2025-01-15     |
| antal  | quantity, qty, försäljning    | 45             |

En exempelfil finns i `sample_data.csv`.

---

## Statusar

| Symbol | Status      | Villkor                  |
|--------|-------------|--------------------------|
| 🔴     | Brådskande  | Lagret räcker < 3 dagar  |
| 🟡     | Snart       | Lagret räcker < 7 dagar  |
| 🟢     | Planera     | Lagret räcker > 7 dagar  |

---

## API-endpoints

| Metod | Endpoint                    | Beskrivning                    |
|-------|-----------------------------|--------------------------------|
| POST  | `/api/login`                | Logga in, returnerar token     |
| POST  | `/api/logout`               | Logga ut                       |
| POST  | `/api/upload`               | Ladda upp CSV/Excel            |
| GET   | `/api/products`             | Lista alla produkter           |
| GET   | `/api/sales/<produkt>`      | Försäljningshistorik           |
| GET   | `/api/forecast/<produkt>`   | 7-dagarsprognos                |
| GET   | `/api/recommendations`      | Alla beställningsrekommendationer |
| PUT   | `/api/stock/<produkt>`      | Uppdatera lagernivå            |
| POST  | `/api/demo`                 | Generera demo-data             |

---

## Projektstruktur

```
grossistprognoser/
├── backend/
│   ├── app.py           # Flask API
│   ├── database.py      # SQLite-setup
│   ├── forecast.py      # Prophet-prognos + fallback
│   ├── requirements.txt
│   └── grossist.db      # Skapas automatiskt vid start
├── frontend/
│   └── index.html       # Dashboard (Tailwind + Chart.js)
├── sample_data.csv      # Exempeldata för test
├── start_backend.bat    # Snabbstart för Windows
└── README.md
```
