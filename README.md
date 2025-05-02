# CPT Soil Modeling Framework

Un framework Python modulare per creare modelli 3D del suolo da dati CPT (Cone Penetration Test) utilizzando tecniche di machine learning.

## Caratteristiche

- **Caricamento dati robusto**: carica dati CPT da vari formati CSV con gestione automatica delle coordinate
- **Visualizzazione dati**: strumenti per esplorare dati CPT e visualizzare profili
- **Classificazione del suolo**: modelli ML per predire tipi di suolo da parametri CPT
- **Interpolazione 3D**: converte punti CPT discreti in un modello di volume 3D continuo
- **Visualizzazione 3D interattiva**: visualizzazione interattiva con Plotly e sezioni trasversali
- **Configurazione flessibile**: sistema YAML per configurare tutti gli aspetti del framework

## Installazione

```bash
# Installa dal repo
git clone https://github.com/geotechnicalteam/cpt_soil_modeling.git
cd cpt_soil_modeling
pip install -e .

# Oppure installa direttamente da PyPI
pip install cpt-soil-modeling
```

## Requisiti

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy
- plotly
- xgboost
- joblib
- pyyaml

## Struttura del progetto

```
cpt_soil_modeling/
├── README.md                  # Questo file
├── requirements.txt           # Dipendenze richieste
├── setup.py                   # Script di installazione
├── config.yaml                # Configurazione predefinita
├── cpt_soil_modeling/         # Package principale
│   ├── __init__.py
│   ├── data/                  # Moduli per caricamento dati
│   │   ├── __init__.py
│   │   ├── loader.py          # Caricatore dati CPT
│   │   └── validator.py       # Validatore dati
│   ├── models/                # Moduli di machine learning 
│   │   ├── __init__.py
│   │   ├── soil_classifier.py # Classificatore di suolo
│   │   └── interpolator.py    # Interpolatore 3D
│   ├── visualization/         # Moduli di visualizzazione
│   │   ├── __init__.py
│   │   ├── cpt_plotter.py     # Visualizzazione dati CPT
│   │   └── model_3d.py        # Visualizzazione modello 3D
│   ├── utils/                 # Utilità
│   │   ├── __init__.py
│   │   ├── exceptions.py      # Eccezioni personalizzate
│   │   ├── config.py          # Gestore configurazione
│   │   └── logger.py          # Sistema di logging
│   └── main.py                # Script principale
├── examples/                  # Esempi di utilizzo
└── tests/                     # Test
```

## Utilizzo base

```python
# Esempio di utilizzo
from cpt_soil_modeling.data.loader import CPTDataLoader
from cpt_soil_modeling.models.soil_classifier import SoilClassifier
from cpt_soil_modeling.models.interpolator import SoilInterpolator
from cpt_soil_modeling.visualization.model_3d import SoilModel3DVisualizer

# Carica i dati
loader = CPTDataLoader()
data = loader.load_data("data/CPT_*.csv")

# Addestra un modello di classificazione
classifier = SoilClassifier()
classifier.train_model(data)
data['predicted_soil'] = classifier.predict(data)

# Crea interpolazione 3D
interpolator = SoilInterpolator()
interpolation = interpolator.create_interpolation(data, 'predicted_soil')

# Visualizza il modello 3D
visualizer = SoilModel3DVisualizer()
visualizer.set_soil_colors(data['predicted_soil'].unique())
visualizer.visualize_3d_model(interpolation, data=data)
```

## Linea di comando

```bash
# Esegui l'intera pipeline
cpt_soil_modeling --data "data/CPT_*.csv" --output "./output"

# Opzioni aggiuntive
cpt_soil_modeling --data "data/CPT_*.csv" \
                  --output "./output" \
                  --config "my_config.yaml" \
                  --model-type "xgb" \
                  --resolution 5 \
                  --interp-method "robust" \
                  --cross-sections
```

## Contribuire

Contributi sono benvenuti! Per favore leggi [CONTRIBUTING.md](CONTRIBUTING.md) per dettagli su come contribuire.

## Licenza

Questo progetto è sotto licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.


1. terreno sensitivo a grana fine;
2. terreno organico, torba;
3. argille: da argille ad argille limose;
4. limi: da limi argillosi a argille limose;
5. sabbie: da sabbie limose a limi sabbiosi;
6. sabbie: da sabbie pulite a sabbie limose;
7. da sabbie ghiaiose a sabbie;
8. da sabbie molto dense a sabbie argillose fortemente sovraconsolidate o cementate;
9. materiali fini granulari molto duri, fortemente sovraconsolidati o cementati.