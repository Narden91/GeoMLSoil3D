import pandas as pd
import numpy as np

class SoilTypeManager:
    """
    Classe per gestire i tipi di suolo, loro codifiche e descrizioni.
    """
    
    # Mappatura per i tipi di suolo secondo la classificazione basata su qc e Rf
    SOIL_TYPES = {
        1: {"abbr": "SGF", "desc": "Terreno sensitivo a grana fine"},
        2: {"abbr": "TOR", "desc": "Terreno organico, torba"},
        3: {"abbr": "ARG", "desc": "Argille: da argille ad argille limose"},
        4: {"abbr": "LIM", "desc": "Limi: da limi argillosi a argille limose"},
        5: {"abbr": "SBL", "desc": "Sabbie: da sabbie limose a limi sabbiosi"},
        6: {"abbr": "SBP", "desc": "Sabbie: da sabbie pulite a sabbie limose"},
        7: {"abbr": "SBG", "desc": "Sabbie ghiaiose a sabbie"},
        8: {"abbr": "SBD", "desc": "Sabbie molto dense/argillose sovraconsolidate/cementate"},
        9: {"abbr": "MFG", "desc": "Materiali fini granulari duri, sovraconsolidati/cementati"}
    }
    
    @classmethod
    def get_abbreviation(cls, soil_type):
        """
        Ottiene l'abbreviazione per un tipo di suolo.
        
        Parameters:
        -----------
        soil_type : int
            Codice numerico del tipo di suolo
            
        Returns:
        --------
        str
            Abbreviazione del tipo di suolo
        """
        if soil_type not in cls.SOIL_TYPES:
            return f"S{soil_type}"
        return cls.SOIL_TYPES[soil_type]["abbr"]
    
    @classmethod
    def get_description(cls, soil_type):
        """
        Ottiene la descrizione completa per un tipo di suolo.
        
        Parameters:
        -----------
        soil_type : int
            Codice numerico del tipo di suolo
            
        Returns:
        --------
        str
            Descrizione completa del tipo di suolo
        """
        if soil_type not in cls.SOIL_TYPES:
            return f"Suolo tipo {soil_type}"
        return cls.SOIL_TYPES[soil_type]["desc"]
    
    @classmethod
    def code_from_qc_rf(cls, qc, rf):
        """
        Determina il tipo di suolo basato su qc (resistenza cono) e Rf (rapporto di attrito)
        secondo il sistema di classificazione standard.
        
        Parameters:
        -----------
        qc : float
            Resistenza del cono in MPa
        rf : float
            Rapporto di attrito in %
            
        Returns:
        --------
        int
            Codice del tipo di suolo
        """
        # Implementazione della classificazione sulla base del grafico fornito
        # Queste sono regole semplificate basate sull'immagine
        
        # Regola 1: Terreno sensitivo a grana fine
        if (rf < 1.0 and qc < 2.0):
            return 1
        
        # Regola 2: Terreno organico, torba
        if (rf > 1.5 and qc < 1.0):
            return 2
            
        # Regola 3: Argille
        if (rf > 4.0 and qc < 1.5):
            return 3
            
        # Regola 4: Limi
        if (3.0 < rf < 5.0 and 1.0 < qc < 3.0):
            return 4
            
        # Regola 5: Sabbie limose
        if (2.0 < rf < 4.0 and 2.0 < qc < 8.0):
            return 5
            
        # Regola 6: Sabbie pulite
        if (1.0 < rf < 2.0 and 5.0 < qc < 20.0):
            return 6
            
        # Regola 7: Sabbie ghiaiose
        if (0.5 < rf < 1.5 and qc > 15.0):
            return 7
            
        # Regola 8: Sabbie dense
        if (1.5 < rf < 3.0 and qc > 10.0):
            return 8
            
        # Regola 9: Materiali fini granulari duri
        if (rf > 3.0 and qc > 10.0):
            return 9
            
        # Valore predefinito: classificare come tipo 4 (limi)
        return 4
    
    @classmethod
    def get_all_types(cls):
        """
        Restituisce tutti i tipi di suolo con le relative descrizioni
        
        Returns:
        --------
        dict
            Dizionario con tutti i tipi di suolo
        """
        return cls.SOIL_TYPES
    
    @classmethod
    def create_label_map(cls):
        """
        Crea una mappa di etichette per la visualizzazione
        
        Returns:
        --------
        dict
            Dizionario che mappa i codici numerici alle etichette (codice + abbreviazione)
        """
        return {k: f"{k} - {v['abbr']}" for k, v in cls.SOIL_TYPES.items()}
    
    @classmethod
    def convert_dataset_labels(cls, df, soil_column='soil []', add_columns=True):
        """
        Converte le etichette numeriche nel dataset in abbrevazioni
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenente i dati CPT
        soil_column : str
            Nome della colonna contenente i tipi di suolo
        add_columns : bool
            Se True, aggiunge colonne per abbreviazione e descrizione
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame con colonne aggiuntive per l'abbreviazione e la descrizione
        """
        if soil_column not in df.columns:
            return df
            
        # Crea nuove colonne se richiesto
        if add_columns:
            df['soil_abbr'] = df[soil_column].apply(cls.get_abbreviation)
            df['soil_desc'] = df[soil_column].apply(cls.get_description)
            
        return df