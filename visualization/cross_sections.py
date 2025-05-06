import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.soil_types import SoilTypeManager


def extract_cross_section(interpolation_data, axis='x', position=None, index=None):
    """
    Estrae una sezione trasversale dal modello 3D interpolato
    
    Parameters:
    -----------
    interpolation_data : dict
        Output dall'interpolazione 3D, contenente X, Y, Z, e values
    axis : str
        Asse lungo il quale estrarre la sezione ('x', 'y' o 'z')
    position : float, optional
        Posizione esatta dove estrarre la sezione lungo l'asse specificato
    index : int, optional
        Indice della slice da estrarre (alternativa a position)
        
    Returns:
    --------
    dict
        Dati della sezione trasversale
    """
    # Estrai i dati della griglia
    if 'grid_data' in interpolation_data:
        X = interpolation_data['grid_data']['X']
        Y = interpolation_data['grid_data']['Y']
        Z = interpolation_data['grid_data']['Z']
        values = interpolation_data['grid_data']['values']
    else:
        X = interpolation_data['X']
        Y = interpolation_data['Y']
        Z = interpolation_data['Z']
        values = interpolation_data['values']
    
    # Determina gli assi della sezione in base all'asse specificato
    if axis.lower() == 'x':
        # Sezione YZ ad un X costante
        if position is not None:
            # Trova l'indice più vicino alla posizione specificata
            x_values = np.unique(X)
            idx = np.abs(x_values - position).argmin()
        elif index is not None:
            # Usa l'indice specificato
            idx = index
        else:
            # Usa il valore centrale di default
            idx = X.shape[0] // 2
        
        # Estrai la sezione
        section_values = values[idx, :, :]
        coord1 = Y[idx, :, :]
        coord2 = Z[idx, :, :]
        axis1_label = 'Y Coordinate (m)'
        axis2_label = 'Depth (m)'
        section_position = X[idx, 0, 0]
        title = f'Sezione YZ a X = {section_position:.2f}m'
        
    elif axis.lower() == 'y':
        # Sezione XZ ad un Y costante
        if position is not None:
            # Trova l'indice più vicino alla posizione specificata
            y_values = np.unique(Y)
            idx = np.abs(y_values - position).argmin()
        elif index is not None:
            # Usa l'indice specificato
            idx = index
        else:
            # Usa il valore centrale di default
            idx = Y.shape[1] // 2
        
        # Estrai la sezione
        section_values = values[:, idx, :]
        coord1 = X[:, idx, :]
        coord2 = Z[:, idx, :]
        axis1_label = 'X Coordinate (m)'
        axis2_label = 'Depth (m)'
        section_position = Y[0, idx, 0]
        title = f'Sezione XZ a Y = {section_position:.2f}m'
        
    elif axis.lower() == 'z':
        # Sezione XY ad un Z costante (profondità)
        if position is not None:
            # Trova l'indice più vicino alla posizione specificata
            z_values = np.unique(Z)
            idx = np.abs(z_values - position).argmin()
        elif index is not None:
            # Usa l'indice specificato
            idx = index
        else:
            # Usa il valore centrale di default
            idx = Z.shape[2] // 2
        
        # Estrai la sezione
        section_values = values[:, :, idx]
        coord1 = X[:, :, idx]
        coord2 = Y[:, :, idx]
        axis1_label = 'X Coordinate (m)'
        axis2_label = 'Y Coordinate (m)'
        section_position = Z[0, 0, idx]
        title = f'Sezione XY a profondità = {section_position:.2f}m'
    
    else:
        raise ValueError(f"Asse non valido: {axis}. Usa 'x', 'y' o 'z'.")
    
    # Restituisci i dati della sezione
    return {
        'values': section_values,
        'coord1': coord1,
        'coord2': coord2,
        'axis1_label': axis1_label,
        'axis2_label': axis2_label,
        'section_position': section_position,
        'title': title,
        'axis': axis
    }


def visualize_cross_section(section_data, soil_types=None, soil_colors=None, 
                           show_contour=True, show_heatmap=True, flip_axis2=True):
    """
    Visualizza una sezione trasversale del modello 3D
    
    Parameters:
    -----------
    section_data : dict
        Dati della sezione trasversale da visualize_cross_section
    soil_types : list, optional
        Lista di tipi di suolo
    soil_colors : dict, optional
        Dizionario che mappa i tipi di suolo ai colori
    show_contour : bool
        Se visualizzare il contorno
    show_heatmap : bool
        Se visualizzare la heatmap
    flip_axis2 : bool
        Se invertire l'asse 2 (tipicamente per la profondità)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura interattiva con la sezione trasversale
    """
    # Estrai i dati della sezione
    values = section_data['values']
    coord1 = section_data['coord1']
    coord2 = section_data['coord2']
    axis1_label = section_data['axis1_label']
    axis2_label = section_data['axis2_label']
    title = section_data['title']
    
    # Crea figura
    fig = go.Figure()
    
    # Crea colormap e etichette
    colorscale, tickvals, ticktext = _create_colormap_and_labels(soil_types, soil_colors)
    
    # Aggiungi heatmap
    if show_heatmap:
        fig.add_trace(go.Heatmap(
            z=values,
            x=coord1[0],
            y=coord2[:, 0],
            colorscale=colorscale,
            colorbar=dict(
                title='Soil Type',
                tickvals=tickvals,
                ticktext=ticktext
            ),
            zmin=min(soil_types) if soil_types else None,
            zmax=max(soil_types) if soil_types else None
        ))
    
    # Aggiungi contorni
    if show_contour:
        fig.add_trace(go.Contour(
            z=values,
            x=coord1[0],
            y=coord2[:, 0],
            colorscale=colorscale,
            contours=dict(
                start=min(soil_types) if soil_types else None,
                end=max(soil_types) if soil_types else None,
                size=1
            ),
            showscale=False,
            line=dict(width=0.5),
            opacity=0.8
        ))
    
    # Aggiorna layout
    fig.update_layout(
        title=title,
        xaxis_title=axis1_label,
        yaxis_title=axis2_label,
        width=800,
        height=600
    )
    
    # Inverti l'asse 2 se richiesto (tipicamente per la profondità)
    if flip_axis2 and section_data['axis'] != 'z':
        fig.update_layout(yaxis_autorange="reversed")
    
    return fig


def visualize_compare_cross_sections(ml_model_data, real_model_data, 
                                   axis='x', position=None, index=None,
                                   soil_types=None, soil_colors=None, show=True):
    """
    Visualizza e confronta sezioni trasversali di due modelli 3D
    
    Parameters:
    -----------
    ml_model_data : dict
        Dati del modello predetto da ML
    real_model_data : dict
        Dati del modello basato sui dati CPT reali
    axis : str
        Asse lungo il quale estrarre la sezione ('x', 'y' o 'z')
    position : float, optional
        Posizione esatta dove estrarre la sezione lungo l'asse specificato
    index : int, optional
        Indice della slice da estrarre (alternativa a position)
    soil_types : list, optional
        Lista di tipi di suolo
    soil_colors : dict, optional
        Dizionario che mappa i tipi di suolo ai colori
    show : bool, optional
        Se mostrare automaticamente la figura
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura interattiva con il confronto delle sezioni
    """
    # Estrai le sezioni trasversali
    ml_section = extract_cross_section(ml_model_data, axis, position, index)
    real_section = extract_cross_section(real_model_data, axis, position, index)
    
    # Crea subplot con due grafici affiancati
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Modello ML (Predetto)', 'Modello Geotecnico (Dati Reali)'),
        shared_yaxes=True
    )
    
    # Crea colormap e etichette
    colorscale, tickvals, ticktext = _create_colormap_and_labels(soil_types, soil_colors)
    
    # Aggiungi le heatmap per entrambi i modelli
    _add_section_to_subplot(fig, ml_section, colorscale, tickvals, ticktext, 1, 1, True)
    _add_section_to_subplot(fig, real_section, colorscale, tickvals, ticktext, 1, 2, False)
    
    # Inverti l'asse Y per la profondità
    if axis != 'z':
        fig.update_layout(yaxis_autorange="reversed")
    
    # Aggiorna il layout complessivo
    fig.update_layout(
        title=f'Confronto Sezioni Trasversali - {ml_section["title"]}',
        width=1200,
        height=600
    )
    
    # Mostra la figura se richiesto
    if show:
        fig.show()
    
    return fig



def _add_section_to_subplot(fig, section_data, colorscale, tickvals, ticktext, row, col, show_colorbar):
    """
    Aggiunge una sezione trasversale a un subplot
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figura a cui aggiungere la sezione
    section_data : dict
        Dati della sezione trasversale
    colorscale : list
        Scala di colori
    tickvals : list
        Valori dei tick per la colorbar
    ticktext : list
        Testo dei tick per la colorbar
    row, col : int
        Riga e colonna del subplot
    show_colorbar : bool
        Se mostrare la colorbar
    """
    # Estrai i dati della sezione
    values = section_data['values']
    coord1 = section_data['coord1']
    coord2 = section_data['coord2']
    
    # Aggiungi heatmap
    fig.add_trace(go.Heatmap(
        z=values,
        x=coord1[0],
        y=coord2[:, 0],
        colorscale=colorscale,
        colorbar=dict(
            title='Soil Type',
            tickvals=tickvals,
            ticktext=ticktext,
            x=0.46 if col == 1 else 0.96,
            len=0.9
        ),
        showscale=show_colorbar
    ), row=row, col=col)
    
    # Aggiungi contorni
    fig.add_trace(go.Contour(
        z=values,
        x=coord1[0],
        y=coord2[:, 0],
        colorscale=colorscale,
        contours=dict(
            size=1
        ),
        showscale=False,
        line=dict(width=0.5),
        opacity=0.8
    ), row=row, col=col)
    
    # Imposta i titoli degli assi
    fig.update_xaxes(title_text=section_data['axis1_label'], row=row, col=col)
    if col == 1:  # Solo per la prima colonna
        fig.update_yaxes(title_text=section_data['axis2_label'], row=row, col=col)


def create_interactive_cross_section_ui(cpt_data, ml_model_data, real_model_data, 
                                       soil_types=None, soil_colors=None, show=True):
    """
    Crea un'interfaccia interattiva per esplorare le sezioni trasversali
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame contenente i dati CPT
    ml_model_data : dict
        Dati del modello predetto da ML
    real_model_data : dict
        Dati del modello basato sui dati CPT reali
    soil_types : list, optional
        Lista di tipi di suolo
    soil_colors : dict, optional
        Dizionario che mappa i tipi di suolo ai colori
    show : bool, optional
        Se mostrare automaticamente la figura
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura interattiva con sliders per navigare le sezioni
    """
    # Estrai i dati della griglia per ottenere le dimensioni
    if 'grid_data' in ml_model_data:
        X = ml_model_data['grid_data']['X']
        Y = ml_model_data['grid_data']['Y']
        Z = ml_model_data['grid_data']['Z']
    else:
        X = ml_model_data['X']
        Y = ml_model_data['Y']
        Z = ml_model_data['Z']
    
    # Ottenere i valori unici per ogni asse
    x_values = np.unique(X)
    y_values = np.unique(Y)
    z_values = np.unique(Z)
    
    # Valori iniziali per le sezioni
    initial_x = x_values[len(x_values) // 2]
    initial_y = y_values[len(y_values) // 2]
    initial_z = z_values[len(z_values) // 2]
    
    # Crea la figura di base con la sezione iniziale (usiamo x come default)
    fig = visualize_compare_cross_sections(
        ml_model_data, real_model_data, 
        axis='x', position=initial_x,
        soil_types=soil_types, soil_colors=soil_colors,
        show=False  # Non mostrare qui per evitare duplicazioni
    )
    
    # Aggiungi CPT locations che intersecano questa sezione
    _add_cpt_locations_to_section(fig, cpt_data, 'x', initial_x, 0.5)
    
    # Crea sliders per ogni asse
    sliders = []
    
    # Slider per l'asse X
    steps_x = []
    for i, x in enumerate(x_values):
        step = dict(
            method="update",
            args=[
                # Qui andrebbero i nuovi dati per ogni frame, ma gestiremo questo con i callback
                {},
                {"title": f"Sezione YZ a X = {x:.2f}m"}
            ],
            label=f"{x:.1f}"
        )
        steps_x.append(step)
    
    sliders.append(dict(
        active=len(x_values) // 2,
        currentvalue={"prefix": "X: "},
        pad={"t": 50},
        steps=steps_x,
        x=0.1,
        xanchor="left",
        y=0,
        yanchor="top",
        len=0.8
    ))
    
    # Slider per l'asse Y
    steps_y = []
    for i, y in enumerate(y_values):
        step = dict(
            method="update",
            args=[
                {},
                {"title": f"Sezione XZ a Y = {y:.2f}m"}
            ],
            label=f"{y:.1f}"
        )
        steps_y.append(step)
    
    sliders.append(dict(
        active=len(y_values) // 2,
        currentvalue={"prefix": "Y: "},
        pad={"t": 50},
        steps=steps_y,
        x=0.1,
        xanchor="left",
        y=0.05,
        yanchor="top",
        len=0.8
    ))
    
    # Slider per l'asse Z (profondità)
    steps_z = []
    for i, z in enumerate(z_values):
        step = dict(
            method="update",
            args=[
                {},
                {"title": f"Sezione XY a profondità = {z:.2f}m"}
            ],
            label=f"{z:.1f}"
        )
        steps_z.append(step)
    
    sliders.append(dict(
        active=len(z_values) // 2,
        currentvalue={"prefix": "Z: "},
        pad={"t": 50},
        steps=steps_z,
        x=0.1,
        xanchor="left",
        y=0.1,
        yanchor="top",
        len=0.8
    ))
    
    # Aggiungi i bottoni per selezionare l'asse
    updatemenus = [
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    label="Sezione YZ (X costante)",
                    method="update",
                    args=[{}, {"title": f"Sezione YZ a X = {initial_x:.2f}m"}]
                ),
                dict(
                    label="Sezione XZ (Y costante)",
                    method="update",
                    args=[{}, {"title": f"Sezione XZ a Y = {initial_y:.2f}m"}]
                ),
                dict(
                    label="Sezione XY (Z costante)",
                    method="update",
                    args=[{}, {"title": f"Sezione XY a profondità = {initial_z:.2f}m"}]
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )
    ]
    
    # Aggiorna il layout con slider e bottoni
    fig.update_layout(
        sliders=sliders,
        updatemenus=updatemenus,
        height=700  # Aumenta l'altezza per fare spazio agli slider
    )
    
    # Mostra la figura se richiesto
    if show:
        fig.show()
    
    return fig


def _add_cpt_locations_to_section(fig, cpt_data, axis, position, tolerance=1.0):
    """
    Aggiunge le posizioni CPT che intersecano una sezione
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figura a cui aggiungere le posizioni CPT
    cpt_data : pandas.DataFrame
        DataFrame contenente i dati CPT
    axis : str
        Asse della sezione ('x', 'y', o 'z')
    position : float
        Posizione della sezione lungo l'asse specificato
    tolerance : float
        Tolleranza per considerare un CPT come intersecante la sezione
    """
    if cpt_data is None or len(cpt_data) == 0:
        return
    
    # Ottieni le coordinate uniche per ogni CPT
    cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
    
    # Filtra i CPT che intersecano la sezione
    if axis.lower() == 'x':
        # Filtra i CPT con x_coord vicino alla posizione
        intersecting_cpts = cpt_locations[np.abs(cpt_locations['x_coord'] - position) <= tolerance]
        
        # Aggiungi marcatori alla prima colonna (ML model)
        if len(intersecting_cpts) > 0:
            fig.add_trace(go.Scatter(
                x=intersecting_cpts['y_coord'],
                y=cpt_data[cpt_data['cpt_id'].isin(intersecting_cpts.index)][cpt_data.columns[0]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='triangle-down'
                ),
                name='CPT Positions',
                showlegend=True
            ), row=1, col=1)
            
            # Aggiungi marcatori alla seconda colonna (Real model)
            fig.add_trace(go.Scatter(
                x=intersecting_cpts['y_coord'],
                y=cpt_data[cpt_data['cpt_id'].isin(intersecting_cpts.index)][cpt_data.columns[0]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='triangle-down'
                ),
                name='CPT Positions',
                showlegend=False
            ), row=1, col=2)
            
    elif axis.lower() == 'y':
        # Filtra i CPT con y_coord vicino alla posizione
        intersecting_cpts = cpt_locations[np.abs(cpt_locations['y_coord'] - position) <= tolerance]
        
        # Aggiungi marcatori alla prima colonna (ML model)
        if len(intersecting_cpts) > 0:
            fig.add_trace(go.Scatter(
                x=intersecting_cpts['x_coord'],
                y=cpt_data[cpt_data['cpt_id'].isin(intersecting_cpts.index)][cpt_data.columns[0]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='triangle-down'
                ),
                name='CPT Positions',
                showlegend=True
            ), row=1, col=1)
            
            # Aggiungi marcatori alla seconda colonna (Real model)
            fig.add_trace(go.Scatter(
                x=intersecting_cpts['x_coord'],
                y=cpt_data[cpt_data['cpt_id'].isin(intersecting_cpts.index)][cpt_data.columns[0]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='triangle-down'
                ),
                name='CPT Positions',
                showlegend=False
            ), row=1, col=2)
            
    elif axis.lower() == 'z':
        # Per la sezione a Z costante, aggiungiamo tutti i CPT
        # poiché tutti attraversano ogni profondità
        
        # Aggiungi marcatori alla prima colonna (ML model)
        fig.add_trace(go.Scatter(
            x=cpt_locations['x_coord'],
            y=cpt_locations['y_coord'],
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol='triangle-down'
            ),
            name='CPT Positions',
            showlegend=True
        ), row=1, col=1)
        
        # Aggiungi marcatori alla seconda colonna (Real model)
        fig.add_trace(go.Scatter(
            x=cpt_locations['x_coord'],
            y=cpt_locations['y_coord'],
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol='triangle-down'
            ),
            name='CPT Positions',
            showlegend=False
        ), row=1, col=2)


def _create_colormap_and_labels(soil_types, soil_colors):
    """
    Crea colormap e labels per la visualizzazione
    
    Parameters:
    -----------
    soil_types : list
        Lista di tipi di suolo
    soil_colors : dict
        Dizionario che mappa i tipi di suolo ai colori
        
    Returns:
    --------
    colorscale, tickvals, ticktext : tuple
        Colorscale, valori tick, e testo tick per la visualizzazione
    """
    if soil_colors is None:
        # Colormap di default
        return 'Viridis', None, None
    
    # Colormap personalizzata
    colorscale = []
    soil_types_sorted = sorted(soil_colors.keys())
    tickvals = soil_types_sorted
    
    # Crea etichette per i tick
    if isinstance(soil_colors[soil_types_sorted[0]], dict):
        # Nuovo formato con label
        ticktext = [soil_colors[soil_type]['label'] for soil_type in soil_types_sorted]
        
        for i, soil_type in enumerate(soil_types_sorted):
            normalized_val = i / (len(soil_types_sorted) - 1)
            colorscale.append([normalized_val, soil_colors[soil_type]['color']])
    else:
        # Vecchio formato
        ticktext = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in soil_types_sorted]
        
        for i, soil_type in enumerate(soil_types_sorted):
            normalized_val = i / (len(soil_types_sorted) - 1)
            colorscale.append([normalized_val, soil_colors[soil_type]])
    
    return colorscale, tickvals, ticktext