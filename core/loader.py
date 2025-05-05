import pandas as pd
import numpy as np
import os
import re
from glob import glob
import random
from utils.soil_types import SoilTypeManager
from utils.helpers import generate_spiral_coords


def load_locations_from_csv(locations_path="location.csv"):
    """
    Carica le coordinate da un file CSV di localizzazione.
    
    Parameters:
    -----------
    locations_path : str
        Percorso al file CSV con le coordinate
        
    Returns:
    --------
    dict
        Dizionario che mappa LogID alle coordinate {LogID: (x, y)}
    """
    try:
        # Verifica se il file esiste
        if not os.path.exists(locations_path):
            # Se il percorso non è trovato, prova a cercarlo nella sottocartella data
            data_path = os.path.join("data", os.path.basename(locations_path))
            if os.path.exists(data_path):
                locations_path = data_path
                print(f"Trovato file delle coordinate in {locations_path}")
            else:
                print(f"File delle coordinate non trovato né in {locations_path} né in {data_path}")
                return {}
        
        # Carica il file delle coordinate
        locations_df = pd.read_csv(locations_path)
        
        # Verifica le colonne necessarie
        required_cols = ['LogID', 'X', 'Y']
        if not all(col in locations_df.columns for col in required_cols):
            print(f"ATTENZIONE: Il file {locations_path} non contiene tutte le colonne necessarie: {required_cols}")
            return {}
        
        # Crea un dizionario che mappa LogID alle coordinate
        locations_map = {}
        for _, row in locations_df.iterrows():
            locations_map[row['LogID']] = (row['X'], row['Y'])
        
        print(f"Caricate {len(locations_map)} coordinate da {locations_path}")
        return locations_map
    
    except Exception as e:
        print(f"Errore nel caricamento del file delle coordinate {locations_path}: {str(e)}")
        return {}
    

def handle_coordinates(df, index, total_files, x_coord_col=None, y_coord_col=None, file_name="", locations_map=None):
    """
    Gestisce le colonne delle coordinate nel dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame di input
    index : int
        Indice del file
    total_files : int
        Numero totale di file
    x_coord_col : str, optional
        Nome della colonna contenente le coordinate X
    y_coord_col : str, optional
        Nome della colonna contenente le coordinate Y
    file_name : str
        Nome del file per logging
    locations_map : dict, optional
        Dizionario che mappa LogID alle coordinate {LogID: (x, y)}
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con coordinate gestite
    """
    # Verifica se le colonne standard esistono già
    if 'x_coord' in df.columns and 'y_coord' in df.columns:
        return df
    
    # Se abbiamo una mappa delle coordinate, proviamo a usarla
    if locations_map is not None:
        cpt_id = extract_cpt_id_from_filename(file_name)
        if cpt_id is not None and cpt_id in locations_map:
            # Abbiamo trovato le coordinate nella mappa
            x, y = locations_map[cpt_id]
            print(f"Usando coordinate da location.csv per {file_name} (LogID: {cpt_id}): ({x}, {y})")
            df['x_coord'] = x
            df['y_coord'] = y
            return df
    
    # Cerca le colonne delle coordinate se non specificate
    if x_coord_col is None or y_coord_col is None:
        x_candidates = [col for col in df.columns if any(term in col.lower() 
                       for term in ['x', 'east', 'long', 'e_', 'x_', 'utm_e'])]
        y_candidates = [col for col in df.columns if any(term in col.lower() 
                       for term in ['y', 'north', 'lat', 'n_', 'y_', 'utm_n'])]
        
        if x_candidates and y_candidates:
            x_coord_col = x_candidates[0]
            y_coord_col = y_candidates[0]
            print(f"Trovate colonne di coordinate: {x_coord_col}, {y_coord_col} per {file_name}")
    
    # Se abbiamo colonne di coordinate nei dati, usale
    if x_coord_col in df.columns and y_coord_col in df.columns:
        print(f"Uso coordinate dalle colonne '{x_coord_col}' e '{y_coord_col}' per {file_name}.")
        df = df.rename(columns={x_coord_col: 'x_coord', y_coord_col: 'y_coord'})
        
        # Controlla se le coordinate sono costanti nel file, come ci si aspetta per un CPT
        if df['x_coord'].nunique() > 1 or df['y_coord'].nunique() > 1:
            print(f"ATTENZIONE: Multiple coordinate (x, y) trovate nel singolo file CPT {file_name}.")
            print(f"Usando le coordinate più comuni.")
            
            # Usa le coordinate più comuni
            x_most_common = df['x_coord'].mode()[0]
            y_most_common = df['y_coord'].mode()[0]
            df['x_coord'] = x_most_common
            df['y_coord'] = y_most_common
    else:
        print(f"Coordinate non trovate per {file_name}. Generazione di coordinate spaziali uniche per questo CPT.")
        
        # Genera coordinate artificiali con pattern a spirale ma con jitter controllato
        x, y = generate_spiral_coords(index, max(100, total_files))
        
        # Aggiungi un piccolo jitter per evitare problemi di interpolazione
        jitter = 0.2  # Variazione minima per evitare punti perfettamente allineati
        x += np.random.uniform(-jitter, jitter)
        y += np.random.uniform(-jitter, jitter)
        
        # Assegna le stesse coordinate generate a tutte le righe
        df['x_coord'] = x
        df['y_coord'] = y
        
        print(f"Assegnate coordinate generate ({x:.2f}, {y:.2f}) a tutti i punti in {file_name}")
    
    return df


def extract_cpt_id_from_filename(file_name):
    """
    Estrae l'ID del CPT dal nome del file.
    
    Parameters:
    -----------
    file_name : str
        Nome del file CPT (es. CPT_983_RAW01.csv)
        
    Returns:
    --------
    int o None
        ID del CPT se trovato, altrimenti None
    """
    # Verifica che il nome del file non sia vuoto
    if not file_name:
        return None
    
    # Rimuovi l'estensione se presente
    if '.' in file_name:
        file_name = file_name.split('.')[0]
    
    # Prima prova a cercare il pattern CPT_XXX_
    match = re.search(r'CPT_(\d+)_', file_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    
    # Prova pattern alternativi come CPT-XXX o CPT.XXX
    alt_patterns = [r'CPT[-.](\d+)', r'(\d+)[-_]CPT', r'CPT.*?(\d{3,4})']
    for pattern in alt_patterns:
        match = re.search(pattern, file_name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    # Se nessun pattern corrisponde, tenta di estrarre qualsiasi sequenza di 3-4 cifre
    digits_match = re.search(r'(\d{3,4})', file_name)
    if digits_match:
        try:
            return int(digits_match.group(1))
        except ValueError:
            pass
    
    print(f"Impossibile estrarre ID CPT dal nome file: {file_name}")
    return None

def detect_csv_format(file_path):
    """
    Rileva automaticamente il formato del file CSV
    
    Parameters:
    -----------
    file_path : str
        Percorso al file CSV
        
    Returns:
    --------
    dict
        Dizionario con i parametri del formato rilevato
    """
    format_params = {
        'encoding': 'utf-8',
        'separator': ',',
        'header': 0,
        'skip_rows': None
    }
    
    # Prova diverse codifiche
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    first_lines = []
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                first_lines = [f.readline() for _ in range(10)]
                if first_lines:
                    format_params['encoding'] = enc
                    break
        except UnicodeDecodeError:
            continue
            
    # Se non abbiamo potuto leggere il file con nessuna codifica, restituisci i parametri predefiniti
    if not first_lines:
        print(f"Attenzione: impossibile determinare la codifica per {file_path}. Si userà utf-8.")
        return format_params
            
    # Rileva il separatore
    separators = [',', ';', '\t', ' ']
    separator_counts = {}
    
    for sep in separators:
        for line in first_lines:
            separator_counts[sep] = separator_counts.get(sep, 0) + line.count(sep)
    
    # Usa il separatore più comune
    if separator_counts:
        most_common_sep = max(separator_counts.items(), key=lambda x: x[1])[0]
        format_params['separator'] = most_common_sep

    # Prova a rilevare se ci sono righe da saltare all'inizio
    # Questo è più complesso, quindi per ora assumiamo che l'intestazione sia nella prima riga
    # Un'analisi più sofisticata potrebbe cercare pattern nelle prime righe
    
    return format_params

def standardize_column_names(df):
    """
    Standardizza i nomi delle colonne in un formato comune
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame di input
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con nomi di colonne standardizzati
    """
    # Definisci pattern per i nomi delle colonne
    column_patterns = {
        'depth [m]': [r'^depth$', r'^z$', r'^elevation$', r'^prof', r'^depth\s*\[', r'^z\s*\['],
        'qc [MPa]': [r'^qc$', r'^q_c$', r'^cone\s*resistance$', r'^qc\s*\[', r'^tip', r'^resistance'],
        'fs [MPa]': [r'^fs$', r'^f_s$', r'^sleeve\s*friction$', r'^fs\s*\[', r'^sleeve'],
        'Rf [%]': [r'^rf$', r'^friction\s*ratio$', r'^f_r$', r'^rf\s*\[', r'^ratio'],
        'u2 [MPa]': [r'^u2$', r'^pore\s*pressure$', r'^pp$', r'^u2\s*\[', r'^pore'],
        'x_coord': [r'^x$', r'^easting$', r'^longitude$', r'^e$', r'^x.*coord', r'^lon', r'^east'],
        'y_coord': [r'^y$', r'^northing$', r'^latitude$', r'^n$', r'^y.*coord', r'^lat', r'^north']
    }
    
    # Crea una mappatura delle colonne
    column_mapping = {}
    for standard_name, patterns in column_patterns.items():
        for col in df.columns:
            if any(re.search(pattern, col, re.IGNORECASE) for pattern in patterns):
                column_mapping[col] = standard_name
                break
    
    # Rinomina le colonne
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Colonne rinominate: {column_mapping}")
    
    return df

def handle_coordinates(df, index, total_files, x_coord_col=None, y_coord_col=None, file_name="", locations_map=None):
    """
    Gestisce le colonne delle coordinate nel dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame di input
    index : int
        Indice del file
    total_files : int
        Numero totale di file
    x_coord_col : str, optional
        Nome della colonna contenente le coordinate X
    y_coord_col : str, optional
        Nome della colonna contenente le coordinate Y
    file_name : str
        Nome del file per logging
    locations_map : dict, optional
        Dizionario che mappa LogID alle coordinate {LogID: (x, y)}
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con coordinate gestite
    """
    # Verifica se le colonne standard esistono già
    if 'x_coord' in df.columns and 'y_coord' in df.columns:
        return df
    
    # Se abbiamo una mappa delle coordinate, proviamo a usarla
    if locations_map is not None:
        cpt_id = extract_cpt_id_from_filename(file_name)
        if cpt_id is not None and cpt_id in locations_map:
            # Abbiamo trovato le coordinate nella mappa
            x, y = locations_map[cpt_id]
            print(f"Usando coordinate da location.csv per {file_name} (LogID: {cpt_id}): ({x}, {y})")
            df['x_coord'] = x
            df['y_coord'] = y
            return df
    
    # Cerca le colonne delle coordinate se non specificate
    if x_coord_col is None or y_coord_col is None:
        x_candidates = [col for col in df.columns if any(term in col.lower() 
                       for term in ['x', 'east', 'long', 'e_', 'x_', 'utm_e'])]
        y_candidates = [col for col in df.columns if any(term in col.lower() 
                       for term in ['y', 'north', 'lat', 'n_', 'y_', 'utm_n'])]
        
        if x_candidates and y_candidates:
            x_coord_col = x_candidates[0]
            y_coord_col = y_candidates[0]
            print(f"Trovate colonne di coordinate: {x_coord_col}, {y_coord_col} per {file_name}")
    
    # Se abbiamo colonne di coordinate nei dati, usale
    if x_coord_col in df.columns and y_coord_col in df.columns:
        print(f"Uso coordinate dalle colonne '{x_coord_col}' e '{y_coord_col}' per {file_name}.")
        df = df.rename(columns={x_coord_col: 'x_coord', y_coord_col: 'y_coord'})
        
        # Controlla se le coordinate sono costanti nel file, come ci si aspetta per un CPT
        if df['x_coord'].nunique() > 1 or df['y_coord'].nunique() > 1:
            print(f"ATTENZIONE: Multiple coordinate (x, y) trovate nel singolo file CPT {file_name}.")
            print(f"Usando le coordinate più comuni.")
            
            # Usa le coordinate più comuni
            x_most_common = df['x_coord'].mode()[0]
            y_most_common = df['y_coord'].mode()[0]
            df['x_coord'] = x_most_common
            df['y_coord'] = y_most_common
    else:
        print(f"Coordinate non trovate per {file_name}. Generazione di coordinate spaziali uniche per questo CPT.")
        
        # Genera coordinate artificiali con pattern a spirale ma con jitter controllato
        x, y = generate_spiral_coords(index, max(100, total_files))
        
        # Aggiungi un piccolo jitter per evitare problemi di interpolazione
        jitter = 0.2  # Variazione minima per evitare punti perfettamente allineati
        x += np.random.uniform(-jitter, jitter)
        y += np.random.uniform(-jitter, jitter)
        
        # Assegna le stesse coordinate generate a tutte le righe
        df['x_coord'] = x
        df['y_coord'] = y
        
        print(f"Assegnate coordinate generate ({x:.2f}, {y:.2f}) a tutti i punti in {file_name}")
    
    return df

def handle_soil_classification(df, file_name=""):
    """
    Gestisce la classificazione del suolo nel dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame di input
    file_name : str
        Nome del file per logging
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con classificazione del suolo
    """
    # Cerca le colonne necessarie per la classificazione
    qc_col = next((col for col in df.columns if 'qc [MPa]' in col), None)
    rf_col = next((col for col in df.columns if 'Rf [%]' in col), None)
    soil_col = next((col for col in df.columns if 'soil []' in col), None)
    
    # Se non abbiamo la colonna del suolo ma abbiamo qc e Rf, calcoliamo i tipi di suolo
    if soil_col is None and qc_col is not None and rf_col is not None:
        print(f"Calcolo dei tipi di suolo per {file_name} basato su qc e Rf")
        df['soil []'] = df.apply(
            lambda row: SoilTypeManager.code_from_qc_rf(row[qc_col], row[rf_col]), 
            axis=1
        )
    
    # Aggiungi abbreviazioni e descrizioni dei tipi di suolo
    if 'soil []' in df.columns:
        df = SoilTypeManager.convert_dataset_labels(df)
    
    return df

def validate_and_clean_data(df, file_name):
    """
    Valida e pulisce i dati CPT
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame di input
    file_name : str
        Nome del file per logging
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame validato e pulito
    """
    # Controlla colonne richieste
    required_cols = ['depth [m]', 'qc [MPa]']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"ATTENZIONE: colonne richieste mancanti in {file_name}: {missing_cols}")
        
        # Prova a inferire la profondità se mancante
        if 'depth [m]' in missing_cols:
            # Prova a usare la prima colonna
            print(f"Assumo che la prima colonna ({df.columns[0]}) sia la profondità")
            df = df.rename(columns={df.columns[0]: 'depth [m]'})
            missing_cols.remove('depth [m]')
        
        # Prova a inferire qc se mancante
        if 'qc [MPa]' in missing_cols:
            # Cerca colonne con 'qc' nel nome
            qc_candidates = [col for col in df.columns if 'qc' in col.lower()]
            if qc_candidates:
                print(f"Usando {qc_candidates[0]} come qc [MPa]")
                df = df.rename(columns={qc_candidates[0]: 'qc [MPa]'})
                missing_cols.remove('qc [MPa]')
            else:
                # Cerca colonne con 'resistance' nel nome
                res_candidates = [col for col in df.columns if 'resist' in col.lower()]
                if res_candidates:
                    print(f"Usando {res_candidates[0]} come qc [MPa]")
                    df = df.rename(columns={res_candidates[0]: 'qc [MPa]'})
                    missing_cols.remove('qc [MPa]')
    
    # Verifica se ci sono ancora colonne mancanti
    if missing_cols:
        print(f"ATTENZIONE: Impossibile inferire tutte le colonne richieste: {missing_cols}")
    
    # Gestione valori mancanti
    missing_values = df.isnull().sum()
    if missing_values.any():
        missing_cols_with_values = [col for col in df.columns if missing_values[col] > 0]
        print(f"Trovati valori mancanti in {file_name}: {missing_cols_with_values}")
        
        # Interpola per colonne numeriche
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col not in ['cpt_id', 'is_train', 'soil []'] and missing_values[col] > 0:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                print(f"  Interpolati valori mancanti in {col}")
    
    # Gestione outlier
    critical_cols = ['qc [MPa]', 'fs [MPa]', 'Rf [%]']
    for col in critical_cols:
        if col in df.columns and len(df) > 10:  # Solo se abbiamo abbastanza dati
            # Calcola Z-scores e identifica outlier
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > 3)
            
            if outliers.any():
                outlier_count = outliers.sum()
                print(f"  Trovati {outlier_count} outlier in {col} per {file_name}")
                
                # Limita gli outlier a 3 deviazioni standard
                cap_high = df[col].mean() + 3 * df[col].std()
                cap_low = df[col].mean() - 3 * df[col].std()
                
                # Per qc e fs, non permettiamo valori negativi
                if col in ['qc [MPa]', 'fs [MPa]']:
                    cap_low = max(0, cap_low)
                    df.loc[df[col] < 0, col] = 0
                
                df.loc[df[col] > cap_high, col] = cap_high
                df.loc[df[col] < cap_low, col] = cap_low
                
                print(f"  Limitati gli outlier in {col}")
    
    return df

def infer_missing_features(df, file_name):
    """
    Tenta di inferire le caratteristiche mancanti dai dati disponibili
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame di input
    file_name : str
        Nome del file per logging
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con caratteristiche inferite
    """
    # Verifica se mancano caratteristiche importanti
    if 'qc [MPa]' in df.columns and 'fs [MPa]' in df.columns and 'Rf [%]' not in df.columns:
        print(f"Calcolo Rf [%] da qc [MPa] e fs [MPa] per {file_name}")
        # Evita divisione per zero
        df['Rf [%]'] = np.where(df['qc [MPa]'] > 0, 
                              100 * df['fs [MPa]'] / df['qc [MPa]'], 
                              0)
    
    if 'qc [MPa]' in df.columns and 'Rf [%]' in df.columns and 'fs [MPa]' not in df.columns:
        print(f"Calcolo fs [MPa] da qc [MPa] e Rf [%] per {file_name}")
        df['fs [MPa]'] = df['qc [MPa]'] * df['Rf [%]'] / 100
    
    return df

def load_and_preprocess_file(file_path, index, total_files, is_train=True, auto_detect=True, 
                            x_coord_col=None, y_coord_col=None, encoding='utf-8', separator=',', 
                            header=0, depth_col=None, locations_map=None):
    """
    Carica e preelabora un singolo file CPT
    
    Parameters:
    -----------
    file_path : str
        Percorso al file
    index : int
        Indice del file nella sequenza
    total_files : int
        Numero totale di file
    is_train : bool
        Flag per indicare se è training data
    auto_detect : bool
        Se rilevare automaticamente il formato CSV
    x_coord_col : str, optional
        Nome della colonna contenente le coordinate X
    y_coord_col : str, optional
        Nome della colonna contenente le coordinate Y
    encoding : str
        Codifica del file da usare
    separator : str
        Carattere separatore CSV
    header : int
        Indice di riga da usare come intestazione
    depth_col : str, optional
        Nome della colonna contenente valori di profondità
    locations_map : dict, optional
        Dizionario che mappa LogID alle coordinate {LogID: (x, y)}
        
    Returns:
    --------
    pandas.DataFrame
        Dati CPT preprocessati
    """
    # Estrai il nome del file senza estensione
    file_name = os.path.basename(file_path).split('.')[0]
    
    try:
        # Rileva il formato se richiesto
        if auto_detect:
            format_params = detect_csv_format(file_path)
            encoding = format_params['encoding']
            separator = format_params['separator']
            header = format_params['header']
            skip_rows = format_params['skip_rows']
        else:
            skip_rows = None
        
        # Carica il CSV
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=separator, 
                           header=header, skiprows=skip_rows)
        except UnicodeDecodeError:
            # Prova codifiche alternative
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            for enc in encodings:
                if enc != encoding:  # Evita di riprovare la stessa codifica
                    try:
                        print(f"Provo codifica alternativa: {enc}")
                        df = pd.read_csv(file_path, encoding=enc, sep=separator, 
                                       header=header, skiprows=skip_rows)
                        encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise ValueError(f"Impossibile decodificare {file_path} con nessuna codifica conosciuta")
        
        print(f"File {file_name} caricato con codifica {encoding}, separatore '{separator}'")
        
        # Standardizza i nomi delle colonne
        df = standardize_column_names(df)
        
        # Gestisci la colonna della profondità
        if depth_col is not None and depth_col in df.columns:
            df = df.rename(columns={depth_col: 'depth [m]'})
        elif 'depth [m]' not in df.columns:
            # Cerca una colonna di profondità o usa la prima colonna
            depth_candidates = [col for col in df.columns if any(term in col.lower() 
                              for term in ['depth', 'z', 'elev', 'prof'])]
            if depth_candidates:
                df = df.rename(columns={depth_candidates[0]: 'depth [m]'})
                print(f"Colonna di profondità identificata: {depth_candidates[0]}")
            else:
                df = df.rename(columns={df.columns[0]: 'depth [m]'})
                print(f"Nessuna colonna di profondità trovata. Assumo che {df.columns[0]} sia la profondità.")
        
        # Aggiungi identificatori
        df['cpt_id'] = file_name
        df['is_train'] = is_train
        
        # Gestisci le coordinate - Usa la mappa delle coordinate se disponibile
        file_basename = os.path.basename(file_path)
        df = handle_coordinates(df, index, total_files, x_coord_col, y_coord_col, file_basename, locations_map)
        
        # Tenta di inferire caratteristiche mancanti
        df = infer_missing_features(df, file_name)
        
        # Gestisci la classificazione del suolo
        df = handle_soil_classification(df, file_name)
        
        # Valida e pulisci i dati
        df = validate_and_clean_data(df, file_name)
        
        return df
        
    except Exception as e:
        print(f"Errore nel caricamento di {file_path}: {str(e)}")
        raise

def load_cpt_files(file_paths, x_coord_col=None, y_coord_col=None, is_train=True,
                  auto_detect=True, encoding='utf-8', separator=',', header=0, depth_col=None, 
                  locations_path="location.csv"):
    """
    Carica i dati CPT dai file specificati
    
    Parameters:
    -----------
    file_paths : list
        Lista dei percorsi file da caricare
    x_coord_col : str, optional
        Nome della colonna contenente le coordinate X
    y_coord_col : str, optional
        Nome della colonna contenente le coordinate Y
    is_train : bool
        Flag per indicare se è training data
    auto_detect : bool
        Se rilevare automaticamente il formato CSV
    encoding : str
        Codifica del file da usare
    separator : str
        Carattere separatore CSV
    header : int
        Indice di riga da usare come intestazione
    depth_col : str, optional
        Nome della colonna contenente valori di profondità
    locations_path : str
        Percorso al file CSV con le coordinate dei CPT
        
    Returns:
    --------
    pandas.DataFrame
        Dati combinati dai file specificati
    """
    # Prima carica la mappa delle coordinate
    locations_map = load_locations_from_csv(locations_path)
    
    dataset_type = "training" if is_train else "testing"
    all_data = []
    total_files = len(file_paths)
    success_count = 0
    
    for i, file_path in enumerate(file_paths):
        try:
            df = load_and_preprocess_file(
                file_path, i, total_files, is_train, auto_detect,
                x_coord_col, y_coord_col, encoding, separator, header, depth_col, locations_map
            )
            
            # Verifica che la colonna is_train sia presente e impostata correttamente
            if 'is_train' not in df.columns:
                df['is_train'] = is_train
            elif df['is_train'].nunique() > 1:
                # Se ci sono valori misti, forza il valore corretto
                print(f"ATTENZIONE: Valori misti nella colonna is_train per {file_path}. Impostazione forzata a {is_train}")
                df['is_train'] = is_train
            
            all_data.append(df)
            success_count += 1
            print(f"Caricati {len(df)} record da {os.path.basename(file_path)} per {dataset_type}")
            
        except Exception as e:
            print(f"Errore nel caricamento di {file_path}: {str(e)}")
            print("Il file verrà saltato, ma l'elaborazione continuerà")
    
    # Combina tutti i dati
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Dataset {dataset_type} combinato contiene {len(combined_data)} record da {success_count}/{total_files} file")
        
        # Verifica che tutti i record abbiano il flag is_train corretto
        if 'is_train' in combined_data.columns:
            wrong_train_count = combined_data[combined_data['is_train'] != is_train].shape[0]
            if wrong_train_count > 0:
                print(f"ATTENZIONE: {wrong_train_count} record hanno un valore errato nella colonna is_train. Correzione...")
                combined_data['is_train'] = is_train
        
        # Controlla problemi di coordinate
        check_coordinates(combined_data, dataset_type)
        
        return combined_data
    else:
        raise ValueError(f"Nessun dato potrebbe essere caricato per il set {dataset_type}")

def check_coordinates(df, dataset_type=""):
    """
    Verifica problemi di coordinate nel dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame di input
    dataset_type : str
        Tipo di dataset per logging
        
    Returns:
    --------
    bool
        True se le coordinate sono sufficienti per l'interpolazione
    """
    # Controlla coordinate uniche
    unique_x = df['x_coord'].nunique()
    unique_y = df['y_coord'].nunique()
    min_required = 2  # Minimo richiesto per l'interpolazione
    
    issues_found = False
    
    if unique_x < min_required:
        print(f"\nATTENZIONE: Solo {unique_x} coordinate x uniche nel set {dataset_type}.")
        print("L'interpolazione 3D potrebbe fallire. Considera di specificare coordinate esplicite.")
        issues_found = True
    
    if unique_y < min_required:
        print(f"\nATTENZIONE: Solo {unique_y} coordinate y uniche nel set {dataset_type}.")
        print("L'interpolazione 3D potrebbe fallire. Considera di specificare coordinate esplicite.")
        issues_found = True
    
    # Controlla la collinearità (tutti i punti su una linea)
    if unique_x >= min_required and unique_y >= min_required:
        cpt_locations = df.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        if len(cpt_locations) >= 3:  # Servono almeno 3 punti
            try:
                correlation = np.abs(np.corrcoef(cpt_locations['x_coord'], cpt_locations['y_coord'])[0, 1])
                if correlation > 0.95:  # Alta correlazione = punti quasi collineari
                    print(f"\nATTENZIONE: Le posizioni CPT sembrano quasi collineari (correlazione: {correlation:.3f}).")
                    print("Questo potrebbe causare problemi con l'interpolazione 3D.")
                    issues_found = True
            except:
                pass
    
    if issues_found:
        print("\nSugggerimenti per risolvere problemi di coordinate:")
        print("1. Specifica manualmente le colonne x_coord_col e y_coord_col nel caricamento dei dati")
        print("2. Assicurati che i file CPT originali contengano coordinate corrette")
        print("3. Se le coordinate sono artificiali, aumenta il jitter per evitare collinearità")
        print("4. Se possibile, aumenta il numero di test CPT a disposizione")
    else:
        print(f"\nLe coordinate nel set {dataset_type} sembrano adeguate per l'interpolazione 3D.")
    
    return not issues_found

def split_cpt_files(file_pattern, test_size=0.2, random_state=42):
    """
    Divide i file CPT in training e testing
    
    Parameters:
    -----------
    file_pattern : str
        Pattern per trovare i file CPT
    test_size : float
        Proporzione di file da usare per testing
    random_state : int
        Seed per riproducibilità
        
    Returns:
    --------
    tuple
        Liste contenenti percorsi file per training e testing
    """
    # Trova tutti i file corrispondenti
    file_paths = glob(file_pattern)
    if not file_paths:
        raise ValueError(f"Nessun file trovato con pattern: {file_pattern}")
    
    total_files = len(file_paths)
    print(f"Trovati {total_files} file CPT")
    
    # Imposta seed per riproducibilità
    random.seed(random_state)
    
    # Mescola i file
    shuffled_files = file_paths.copy()
    random.shuffle(shuffled_files)
    
    # Calcola numero di file test
    n_test = max(1, int(len(shuffled_files) * test_size))
    
    # Divide i file
    test_files = shuffled_files[:n_test]
    train_files = shuffled_files[n_test:]
    
    print(f"Divisi in {len(train_files)} file training e {len(test_files)} file testing")
    
    return train_files, test_files