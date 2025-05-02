import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import local modules
from cpt_model import CPT_3D_SoilModel
from utils import debug_cpt_files


def main():
    """
    Main function to run the CPT 3D Soil Model framework
    """
    # Create framework instance
    framework = CPT_3D_SoilModel()
    
    # Analisi preliminare dei file CPT (opzionale)
    debug_cpt_files("data/CPT_*.csv")
    
    # 1. Load data con coordinate generate in modo più robusto
    data = framework.load_data("data/CPT_*.csv")
    
    # Verifica che ci siano sufficienti coordinate uniche
    x_unique = framework.cpt_data['x_coord'].nunique()
    y_unique = framework.cpt_data['y_coord'].nunique()
    
    print(f"Dataset has {x_unique} unique x-coordinates and {y_unique} unique y-coordinates")
    
    if x_unique < 2 or y_unique < 2:
        print("WARNING: Not enough unique coordinates for 3D interpolation.")
        print("Consider adding proper spatial coordinates to your CPT files.")
    
    # 2. Explore data
    framework.explore_data()
    
    # 3. Train soil classification model
    model = framework.train_soil_classification_model(model_type='rf')
    
    # 4. Predict soil types for all data points
    predictions = framework.predict_soil_types()
    
    # 5. Create 3D interpolation - provare prima il metodo alternativo
    try:
        print("Trying alternative interpolation method...")
        interpolation = framework.create_3d_interpolation_alternative(resolution=5)
    except Exception as e:
        print(f"Alternative interpolation failed: {e}")
        print("Falling back to standard interpolation with robustness improvements...")
        try:
            interpolation = framework.create_3d_interpolation(resolution=5)
        except Exception as e:
            print(f"3D interpolation failed again: {e}")
            print("Creating simplified 2D visualization instead...")
            # Qui potresti aggiungere una funzione di visualizzazione 2D semplificata
    
    # 6. Visualize 3D model
    try:
        framework.visualize_3d_model(interpolation, interactive=True)
    except Exception as e:
        print(f"3D visualization failed: {e}")
        print("Consider visualizing individual CPT profiles instead")
    
    # 7. Save model for future use
    framework.save_model('cpt_soil_model.pkl')


if __name__ == "__main__":
    main()