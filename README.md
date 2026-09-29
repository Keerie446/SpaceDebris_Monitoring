Space Debris Risk Monitor
========================

End-to-end space debris risk visualization and collision risk "zone" dashboard.

## How to run

1. Install dependencies

```bash
cd "/Users/keerthana/Projects/Space_debris"
python3 -m pip install -r requirements.txt
```

2. Start the app

```bash
streamlit run streamlit_app.py
```

## Dataset

The app expects a user-uploaded dataset (`CSV` or `Parquet`). Column names can vary; use the column-mapping step in the UI.

### Supported core fields (recommended)
- `Altitude` (km)
- `Velocity` (km/s or m/s)
- `Inclination` (deg)
- `Object size` (any consistent numeric unit)
- `Orbit type` (LEO/MEO/GEO or something mappable)
- `Launch year` (year as int or parseable date)
- `Decay status` (decayed/not-decayed or boolean-like)

If `Velocity` or `Object size` is missing, the app uses safe defaults (velocity can be estimated from altitude).

