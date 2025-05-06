# 🌍 CPT Soil Modeling Framework

A Python framework for creating 3D soil models from CPT (Cone Penetration Test) data using machine learning techniques.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

## 📋 Overview

CPT Soil Modeling Framework is a comprehensive tool for geotechnical engineers and civil engineers that transforms Cone Penetration Test (CPT) data into three-dimensional soil models, facilitating the analysis and visualization of subsurface soil characteristics. The framework combines machine learning techniques for soil classification with spatial interpolation to create accurate 3D representations of subsurface conditions.

## ✨ Key Features

- 📊 **Robust data loading**: supports various CSV formats with automatic column detection
- 🧠 **ML soil classification**: uses Random Forest or XGBoost to predict soil types
- 🌐 **3D interpolation**: converts discrete CPT points into a continuous volumetric model
- 📈 **Interactive visualizations**: explore models with Plotly and dynamic cross-sections
- 🏗️ **Foundation analysis**: recommendations for construction techniques based on soil composition
- ⚙️ **Flexible configuration**: YAML system to customize all aspects of the framework

## 📦 Requirements

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

## 🚀 Quick Start Guide

### Command Line Usage

The simplest way to use the framework is through the command line:

```bash
# Basic execution with default settings
python main.py --data "data/CPT_*.csv" --output "./output"

# Execution with advanced options
python main.py --data "data/CPT_*.csv" \
               --output "./output" \
               --config "my_config.yaml" \
               --model-type "xgb" \
               --resolution 5 \
               --cross-sections \
               --analyze-foundation True
```

### Python Script Usage

```python
from core.model import CPT_3D_SoilModel

# Initialize the framework
model = CPT_3D_SoilModel()

# Load data
model.load_data("data/CPT_*.csv")

# Train the classification model
model.train_soil_classification_model(model_type="rf")

# Predict soil types
model.predict_soil_types()

# Create 3D model
interp_data = model.create_3d_interpolation(resolution=5)

# Visualize the 3D model
model.visualize_3d_model(interp_data, interactive=True)

# Analyze foundations
from core.foundation_analysis import analyze_foundation_compatibility
analyzer = analyze_foundation_compatibility(model.cpt_data, model.ml_model_data)
analyzer.visualize_recommendations()
```

## ⚙️ Configuration

The framework uses a YAML configuration file to control all aspects of processing. The default `config.yaml` file can be modified to customize:

- 📁 Input/output paths
- 🔍 Data loading parameters
- 🧮 Machine learning model settings
- 🌐 3D interpolation parameters
- 🎨 Visualization options
- 🏗️ Foundation analysis settings

Example configuration:

```yaml
paths:
  data: "data/CPT_*.csv"
  output_dir: "output/"

soil_classification:
  model_type: "rf"  # Random Forest
  test_size: 0.2
  random_state: 42
  use_all_features: true

interpolation:
  resolution: 5
  enabled: true
  
display:
  interactive_visualization: true
  cross_sections: true
  
foundation_analysis:
  enabled: true
  max_depth: 3.0
```

## 🗃️ Supported Soil Types

The framework classifies soil into 9 standard categories:

1. 🧱 **SGF**: Sensitive fine-grained soil
2. 🌱 **TOR**: Organic soil, peat
3. 🔷 **ARG**: Clays: from clays to silty clays
4. 🔶 **LIM**: Silts: from clayey silts to silty clays
5. 🟨 **SBL**: Sands: from silty sands to sandy silts
6. 🟧 **SBP**: Sands: from clean sands to silty sands
7. 🟫 **SBG**: Gravelly sands to sands
8. 🟥 **SBD**: Very dense/clayey sands - overconsolidated/cemented
9. 🟪 **MFG**: Hard fine-grained materials, overconsolidated/cemented

## 🧰 Foundation Analysis

The framework includes an advanced foundation analysis module that:

- Analyzes soil composition up to a specified depth
- Calculates percentages of clay, silt, sand, and gravel
- Recommends optimal construction techniques (Adobe, Pisé, Cob, BTC, Torchis)
- Provides detailed foundation requirements based on soil type
- Visually displays the recommended foundation composition

To activate this feature:

```bash
python main.py --data "data/CPT_*.csv" --analyze-foundation True --foundation-depth 1.5
```

## 📊 Visualizations

The framework offers various visualizations:

- **CPT Profiles**: plots of cone resistance and friction ratio measurements
- **CPT Locations**: spatial map of test points
- **3D Model**: interactive visualization of soil volume
- **Cross-Sections**: 2D views through the 3D model
- **Interactive Panel**: interface for dynamically exploring the model

## 📄 License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 👥 Contact

For any questions or suggestions about the framework, don't hesitate to contact us or open an issue on GitHub.

---