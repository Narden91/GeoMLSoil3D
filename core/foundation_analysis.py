import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

class FoundationAnalyzer:
    """
    Classe per l'analisi della composizione del terreno e la raccomandazione
    delle tecniche costruttive più adatte in terra cruda
    """
    
    # Definizioni dei range ottimali per le diverse tecniche costruttive
    # formato: (min_argilla, max_argilla, min_limo, max_limo, min_sabbia, max_sabbia, note)
    CONSTRUCTION_TECHNIQUES = {
        'Adobe': (15, 25, 25, 40, 40, 60, 'Richiede una buona coesione; l\'aggiunta di fibre vegetali può migliorare la resistenza'),
        'Pisé': (5, 15, 20, 35, 50, 70, 'Necessità di una compattazione efficace; una bassa percentuale di argilla è sufficiente'),
        'Cob': (15, 25, 20, 30, 50, 60, 'L\'aggiunta di fibre (es. paglia) è essenziale per aumentare la resistenza alla trazione'),
        'BTC': (10, 20, 15, 25, 55, 65, 'Spesso stabilizzati con il 5-10% di leganti idraulici (es. calce o cemento) per migliorare la durabilità'),
        'Torchis': (10, 20, 20, 30, 50, 60, 'Utilizza una struttura di supporto in legno o canne; l\'aggiunta di fibre è fondamentale')
    }
    
    # Requisiti per le fondamenta in base al tipo di suolo
    FOUNDATION_REQUIREMENTS = {
        'Argilloso': {
            'profondità_minima': 80,  # cm
            'larghezza_minima': 60,   # cm
            'base_drenante': 20,      # cm
            'composizione': 'Strato drenante di ghiaia (20cm) + Stabilizzazione con calce (5-8%) + Isolamento dall\'umidità',
            'note': 'Terreni argillosi sono soggetti a rigonfiamento e ritiro; fondazioni più profonde e drenaggio efficace sono essenziali'
        },
        'Limoso': {
            'profondità_minima': 60,  # cm
            'larghezza_minima': 50,   # cm
            'base_drenante': 15,      # cm
            'composizione': 'Strato drenante di ghiaia (15cm) + Geotessile + Stabilizzazione con cemento (4-6%)',
            'note': 'Terreni limosi possono essere instabili con l\'acqua; fornire un buon drenaggio'
        },
        'Sabbioso': {
            'profondità_minima': 50,  # cm
            'larghezza_minima': 40,   # cm
            'base_drenante': 10,      # cm
            'composizione': 'Strato compattato di sabbia e ghiaia (10cm) + Leggera stabilizzazione con cemento (3-5%)',
            'note': 'Terreni sabbiosi offrono buon drenaggio ma minore coesione; compattazione importante'
        },
        'Ghiaioso': {
            'profondità_minima': 40,  # cm
            'larghezza_minima': 40,   # cm
            'base_drenante': 5,       # cm
            'composizione': 'Strato di livellamento + Compattazione + Minima stabilizzazione necessaria',
            'note': 'Terreni ghiaiosi offrono eccellente capacità portante; verificare omogeneità'
        }
    }
    
    def __init__(self, cpt_data, soil_model_data):
        """
        Inizializza l'analizzatore con i dati CPT e il modello del suolo
        
        Parameters:
        -----------
        cpt_data : pandas.DataFrame
            DataFrame contenente i dati CPT
        soil_model_data : dict
            Dati del modello del suolo 3D
        """
        self.cpt_data = cpt_data
        self.soil_model_data = soil_model_data
        self.composition = None
        self.recommendations = None
        self.foundation_type = None
    
    def analyze_soil_composition(self, max_depth=3.0, foundation_depth=1.5):
        """
        Analizza la composizione del terreno fino alla profondità specificata
        
        Parameters:
        -----------
        max_depth : float
            Profondità massima da analizzare (m)
        foundation_depth : float
            Profondità tipica delle fondazioni (m)
            
        Returns:
        --------
        dict
            Composizione stimata del terreno (percentuali di argilla, limo, sabbia, ghiaia)
        """
        print(f"Analisi della composizione del suolo fino a {max_depth}m di profondità...")
        
        # Estrai i dati fino alla profondità massima
        depth_col = self.cpt_data.columns[0]
        shallow_data = self.cpt_data[self.cpt_data[depth_col] <= max_depth]
        
        # Estrai i dati fino alla profondità delle fondazioni
        foundation_data = self.cpt_data[self.cpt_data[depth_col] <= foundation_depth]
        
        # Se abbiamo dati sui tipi di suolo, li usiamo per stimare la composizione
        if 'soil []' in shallow_data.columns:
            self.composition = self._estimate_composition_from_soil_types(shallow_data)
        else:
            # Se non abbiamo classificazione del suolo, cerchiamo di stimare
            # dalla resistenza del cono e dal rapporto di attrito
            self.composition = self._estimate_composition_from_cpt(shallow_data)
        
        print(f"Composizione stimata del terreno: {self.composition}")
        
        # Determina il tipo predominante di suolo per le fondazioni
        self.foundation_type = self._determine_foundation_type()
        
        return self.composition
    
    def _estimate_composition_from_soil_types(self, data):
        """
        Stima la composizione del terreno in base ai tipi di suolo
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame contenente i dati CPT con classificazione del suolo
            
        Returns:
        --------
        dict
            Composizione stimata (percentuali di argilla, limo, sabbia, ghiaia)
        """
        # Conta i diversi tipi di suolo
        soil_counts = data['soil []'].value_counts(normalize=True) * 100
        
        # Converti i tipi di suolo in componenti granulometriche stimate
        # Mappa approssimativa da tipi di suolo a componenti
        soil_composition = {
            1: {'argilla': 40, 'limo': 40, 'sabbia': 15, 'ghiaia': 5},    # Terreno sensitivo a grana fine
            2: {'argilla': 60, 'limo': 30, 'sabbia': 10, 'ghiaia': 0},    # Terreno organico, torba
            3: {'argilla': 70, 'limo': 20, 'sabbia': 10, 'ghiaia': 0},    # Argille
            4: {'argilla': 30, 'limo': 60, 'sabbia': 10, 'ghiaia': 0},    # Limi
            5: {'argilla': 10, 'limo': 30, 'sabbia': 60, 'ghiaia': 0},    # Sabbie limose
            6: {'argilla': 5, 'limo': 15, 'sabbia': 80, 'ghiaia': 0},     # Sabbie pulite
            7: {'argilla': 5, 'limo': 10, 'sabbia': 70, 'ghiaia': 15},    # Sabbie ghiaiose
            8: {'argilla': 20, 'limo': 10, 'sabbia': 65, 'ghiaia': 5},    # Sabbie dense
            9: {'argilla': 25, 'limo': 35, 'sabbia': 35, 'ghiaia': 5}     # Materiali fini granulari duri
        }
        
        # Calcola la composizione media ponderata
        composition = {'argilla': 0, 'limo': 0, 'sabbia': 0, 'ghiaia': 0}
        
        for soil_type, percentage in soil_counts.items():
            if soil_type in soil_composition:
                for component, value in soil_composition[soil_type].items():
                    composition[component] += value * percentage / 100
        
        # Normalizza la composizione a 100%
        total = sum(composition.values())
        for component in composition:
            composition[component] = round(composition[component] / total * 100, 1)
        
        return composition
    
    def _estimate_composition_from_cpt(self, data):
        """
        Stima la composizione del terreno in base ai parametri CPT
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame contenente i dati CPT
            
        Returns:
        --------
        dict
            Composizione stimata (percentuali di argilla, limo, sabbia, ghiaia)
        """
        # Stima basata su qc e Rf
        # Questo è un metodo semplificato e approssimativo
        qc_mean = data['qc [MPa]'].mean()
        rf_mean = data['Rf [%]'].mean() if 'Rf [%]' in data.columns else 0
        
        # Stima approssimativa:
        # - Alta qc, basso Rf = materiali granulari (sabbia, ghiaia)
        # - Bassa qc, alto Rf = materiali coesivi (argilla, limo)
        
        # Inizializza una composizione di default
        composition = {'argilla': 25, 'limo': 30, 'sabbia': 40, 'ghiaia': 5}
        
        # Regola in base a qc
        if qc_mean > 15:  # Alta resistenza
            composition['sabbia'] += 20
            composition['ghiaia'] += 10
            composition['argilla'] -= 15
            composition['limo'] -= 15
        elif qc_mean < 2:  # Bassa resistenza
            composition['argilla'] += 15
            composition['limo'] += 10
            composition['sabbia'] -= 20
            composition['ghiaia'] -= 5
        
        # Regola in base a Rf
        if rf_mean > 4:  # Alto rapporto di attrito
            composition['argilla'] += 10
            composition['limo'] += 5
            composition['sabbia'] -= 10
            composition['ghiaia'] -= 5
        elif rf_mean < 1:  # Basso rapporto di attrito
            composition['sabbia'] += 15
            composition['ghiaia'] += 5
            composition['argilla'] -= 15
            composition['limo'] -= 5
        
        # Assicura valori non negativi e normalizza
        for component in composition:
            composition[component] = max(0, composition[component])
        
        total = sum(composition.values())
        for component in composition:
            composition[component] = round(composition[component] / total * 100, 1)
        
        return composition
    
    def _determine_foundation_type(self):
        """
        Determina il tipo predominante di suolo per le fondazioni
        
        Returns:
        --------
        str
            Tipo di suolo predominante per le fondazioni
        """
        max_component = max(self.composition, key=self.composition.get)
        
        # Mappa componente principale al tipo di suolo
        if max_component == 'argilla':
            return 'Argilloso'
        elif max_component == 'limo':
            return 'Limoso'
        elif max_component == 'sabbia':
            return 'Sabbioso'
        elif max_component == 'ghiaia':
            return 'Ghiaioso'
        else:
            return 'Misto'
    
    def recommend_construction_technique(self):
        """
        Raccomanda la tecnica costruttiva più adatta in base alla composizione del suolo
        
        Returns:
        --------
        dict
            Raccomandazioni per le tecniche costruttive
        """
        if self.composition is None:
            raise ValueError("È necessario analizzare la composizione del suolo prima di fornire raccomandazioni")
        
        print("Analisi delle tecniche costruttive ottimali...")
        
        # Calcola la compatibilità con ogni tecnica costruttiva
        technique_scores = {}
        
        for technique, params in self.CONSTRUCTION_TECHNIQUES.items():
            min_clay, max_clay, min_silt, max_silt, min_sand, max_sand, note = params
            
            clay_score = self._calculate_range_score(self.composition['argilla'], min_clay, max_clay)
            silt_score = self._calculate_range_score(self.composition['limo'], min_silt, max_silt)
            sand_score = self._calculate_range_score(self.composition['sabbia'], min_sand, max_sand)
            
            # Calcola uno score complessivo (semplice media)
            overall_score = (clay_score + silt_score + sand_score) / 3
            
            # Aggiungiamo anche fattori che potrebbero influenzare la scelta
            adjustments = []
            
            # Per terreni con alta percentuale di argilla
            if self.composition['argilla'] > 30:
                if technique in ['Adobe', 'Cob']:
                    overall_score *= 0.8
                    adjustments.append("Alta % di argilla: considera l'aggiunta di sabbia")
            
            # Per terreni con alta percentuale di sabbia
            if self.composition['sabbia'] > 70:
                if technique in ['Adobe', 'Cob']:
                    overall_score *= 0.8
                    adjustments.append("Alta % di sabbia: considera l'aggiunta di più argilla")
                if technique in ['Pisé', 'BTC']:
                    overall_score *= 1.1
                    adjustments.append("Buona % di sabbia per questa tecnica")
            
            # Per terreni con presenza di ghiaia
            if self.composition['ghiaia'] > 10:
                if technique in ['Adobe', 'Torchis']:
                    overall_score *= 0.8
                    adjustments.append("Presenza di ghiaia: potrebbe essere necessario setacciare")
                if technique in ['Pisé', 'BTC']:
                    overall_score *= 1.05
                    adjustments.append("La ghiaia può migliorare la resistenza se ben distribuita")
            
            technique_scores[technique] = {
                'score': round(overall_score * 100, 1),
                'adjustments': adjustments,
                'note': note
            }
        
        # Ordina le tecniche per punteggio
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Prepara le raccomandazioni
        recommendations = {
            'best_technique': sorted_techniques[0][0],
            'best_score': sorted_techniques[0][1]['score'],
            'all_techniques': {name: data for name, data in sorted_techniques},
            'soil_composition': self.composition,
            'foundation_type': self.foundation_type,
            'foundation_requirements': self.FOUNDATION_REQUIREMENTS[self.foundation_type]
        }
        
        self.recommendations = recommendations
        return recommendations
    
    def _calculate_range_score(self, value, min_val, max_val):
        """
        Calcola un punteggio in base alla vicinanza di un valore a un intervallo ottimale
        
        Parameters:
        -----------
        value : float
            Valore da valutare
        min_val : float
            Valore minimo dell'intervallo ottimale
        max_val : float
            Valore massimo dell'intervallo ottimale
            
        Returns:
        --------
        float
            Punteggio tra 0 e 1, dove 1 indica che il valore è nell'intervallo ottimale
        """
        # Se il valore è nell'intervallo ottimale
        if min_val <= value <= max_val:
            return 1.0
        
        # Se il valore è fuori dall'intervallo, calcola la distanza
        if value < min_val:
            distance = min_val - value
            range_size = max_val - min_val
            # Penalità tanto maggiore quanto più ci si allontana dal range
            # Usiamo una funzione che decresce più rapidamente
            return max(0, 1 - (distance / (range_size * 1.5)) ** 1.5)
        else:  # value > max_val
            distance = value - max_val
            range_size = max_val - min_val
            return max(0, 1 - (distance / (range_size * 1.5)) ** 1.5)
    
    def visualize_recommendations(self):
        """
        Visualizza le raccomandazioni con grafici matplotlib
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figura con la visualizzazione delle raccomandazioni
        """
        if self.recommendations is None:
            raise ValueError("È necessario generare raccomandazioni prima di visualizzarle")
        
        # Crea una figura con due sottografici
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Grafico 1: Composizione del suolo
        labels = ['Argilla', 'Limo', 'Sabbia', 'Ghiaia']
        sizes = [self.composition['argilla'], self.composition['limo'], 
                self.composition['sabbia'], self.composition['ghiaia']]
        colors = ['#8B4513', '#D2B48C', '#F4A460', '#BEBEBE']
        explode = (0.1, 0, 0, 0)  # Evidenzia l'argilla
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title('Composizione stimata del terreno')
        
        # Grafico 2: Valutazione tecniche costruttive
        techniques = list(self.recommendations['all_techniques'].keys())
        scores = [self.recommendations['all_techniques'][t]['score'] for t in techniques]
        
        # Ordina le tecniche per punteggio
        sorted_indices = np.argsort(scores)
        techniques = [techniques[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Crea un grafico a barre orizzontale
        bars = ax2.barh(techniques, scores, color='skyblue')
        
        # Evidenzia la tecnica migliore
        best_idx = techniques.index(self.recommendations['best_technique'])
        bars[best_idx].set_color('green')
        
        # Aggiungi etichette ai valori
        for i, v in enumerate(scores):
            ax2.text(v + 1, i, f"{v}%", va='center')
        
        ax2.set_xlabel('Punteggio di compatibilità (%)')
        ax2.set_title('Valutazione delle tecniche costruttive')
        ax2.set_xlim(0, 105)
        
        # Aggiusta il layout
        plt.tight_layout()
        
        # Aggiungi un titolo generale
        plt.suptitle(f"Analisi del terreno e raccomandazioni costruttive", fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def visualize_foundation(self):
        """
        Visualizza la composizione raccomandata per le fondamenta
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figura con la visualizzazione delle fondamenta
        """
        if self.recommendations is None:
            raise ValueError("È necessario generare raccomandazioni prima di visualizzare le fondamenta")
        
        # Estrai i dati delle fondamenta
        foundation_data = self.recommendations['foundation_requirements']
        
        # Crea una figura più ampia per evitare sovrapposizioni
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Definisci le dimensioni
        width = foundation_data['larghezza_minima'] / 100  # Converti in metri
        depth = foundation_data['profondità_minima'] / 100  # Converti in metri
        drainage_height = foundation_data['base_drenante'] / 100  # Converti in metri
        
        # Definisci colori per i diversi strati
        cmap = ListedColormap(['#BEBEBE', '#8B4513', '#D2B48C', '#F4A460'])
        
        # Disegna il profilo del terreno
        # X da -1.5 a +2.5 metri (in orizzontale) per dare più spazio
        # Y da 0 a -2 metri (in verticale)
        x = np.linspace(-1.5, 2.5, 100)
        # Superficie leggermente irregolare
        y_surface = 0.05 * np.sin(2 * np.pi * x / 1.5)
        
        # Disegna la superficie del terreno
        ax.plot(x, y_surface, 'k-', linewidth=2)
        
        # Riempi il terreno sottostante
        ax.fill_between(x, y_surface, -2, color='#D2B48C')
        
        # Disegna lo scavo per la fondazione
        foundation_x_min = 0
        foundation_x_max = width
        foundation_y = -depth
        
        # Disegna le linee dello scavo
        ax.plot([foundation_x_min, foundation_x_min], [y_surface[len(x)//3], foundation_y], 'k--')
        ax.plot([foundation_x_max, foundation_x_max], [y_surface[len(x)//3+30], foundation_y], 'k--')
        ax.plot([foundation_x_min, foundation_x_max], [foundation_y, foundation_y], 'k--')
        
        # Disegna lo strato drenante
        drainage_y = foundation_y + drainage_height
        ax.add_patch(Rectangle((foundation_x_min, foundation_y), width, drainage_height, 
                              edgecolor='black', facecolor='#BEBEBE', alpha=0.7))
        
        # Disegna la fondazione principale
        ax.add_patch(Rectangle((foundation_x_min, drainage_y), width, depth - drainage_height, 
                              edgecolor='black', facecolor='#8B4513', alpha=0.8))
        
        # Disegna l'inizio della struttura sopra la fondazione
        structure_height = 0.3
        structure_width = width * 0.8
        structure_x = foundation_x_min + (width - structure_width) / 2
        ax.add_patch(Rectangle((structure_x, 0), structure_width, structure_height, 
                              edgecolor='black', facecolor='#D2691E', alpha=0.9))
        
        # Aggiungi testo descrittivo
        # Titolo principale
        ax.set_title(f"Composizione raccomandata delle fondamenta - Terreno {self.foundation_type}", fontsize=14)
        
        # Informazioni sulla fondazione - spostato a destra per evitare sovrapposizioni
        foundation_text = (
            f"Requisiti fondazione:\n"
            f"• Profondità minima: {foundation_data['profondità_minima']} cm\n"
            f"• Larghezza minima: {foundation_data['larghezza_minima']} cm\n"
            f"• Base drenante: {foundation_data['base_drenante']} cm\n\n"
            f"Composizione raccomandata:\n{foundation_data['composizione']}\n\n"
            f"Note: {foundation_data['note']}"
        )
        
        # Posiziona il testo delle informazioni - spostato più a destra
        ax.text(1.3, -0.5, foundation_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Etichette per gli strati - riposizionate per evitare sovrapposizioni
        # Posiziona l'etichetta dello strato drenante più in basso e centrato
        ax.text(foundation_x_min + width/2, foundation_y + drainage_height/3, 
               f"Strato drenante\n({int(drainage_height*100)} cm)", 
               ha='center', va='center', fontsize=9,
               bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
        
        # Posiziona l'etichetta della fondazione più al centro
        foundation_text_y = drainage_y + (depth - drainage_height)/2  # Centra verticalmente
        ax.text(foundation_x_min + width/2, foundation_text_y, 
               f"Fondazione\n({int(depth*100-drainage_height*100)} cm)", 
               ha='center', va='center', fontsize=9,
               bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
        
        # Segnala la profondità totale - Migliorato il posizionamento della freccia
        ax.annotate(f"{int(depth*100)} cm", 
                   xy=(foundation_x_max + 0.05, foundation_y), 
                   xytext=(foundation_x_max + 0.3, foundation_y/2),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Segnala la larghezza - Migliorato il posizionamento
        ax.annotate(f"{int(width*100)} cm", 
                   xy=(foundation_x_min + width/2, foundation_y - 0.05), 
                   xytext=(foundation_x_min + width/2, foundation_y - 0.25),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                   fontsize=10, ha='center',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Aggiungi informazioni sulla tecnica costruttiva più adatta
        # Posizionato più in basso per evitare sovrapposizioni
        best_technique = self.recommendations['best_technique']
        best_score = self.recommendations['best_score']
        best_note = self.CONSTRUCTION_TECHNIQUES[best_technique][6]
        
        construction_text = (
            f"Tecnica costruttiva consigliata: {best_technique}\n"
            f"Compatibilità: {best_score}%\n"
            f"Note: {best_note}"
        )
        
        # Posiziona il testo della tecnica costruttiva - più in basso
        ax.text(0.5, -1.8, construction_text, fontsize=10, 
               bbox=dict(facecolor='lightgreen', alpha=0.8), ha='center')
        
        # Composizione del suolo come testo - spostato più a sinistra
        soil_text = "Composizione del suolo:\n"
        for component, value in self.composition.items():
            soil_text += f"• {component.capitalize()}: {value}%\n"
        
        # Posiziona il testo della composizione
        ax.text(-1.3, -0.5, soil_text, fontsize=10, 
              bbox=dict(facecolor='#F0E68C', alpha=0.8))
        
        # Imposta i limiti degli assi allargati per dare più spazio
        ax.set_xlim(-1.5, 2.7)
        ax.set_ylim(-2, 0.5)
        ax.set_xlabel('Larghezza (m)')
        ax.set_ylabel('Profondità (m)')
        
        # Assicura la corretta proporzione
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Assicurati che il grafico venga mostrato
        plt.show()
        
        return fig


def analyze_foundation_compatibility(cpt_data, soil_model_data, max_depth=3.0):
    """
    Analizza la compatibilità del terreno con le diverse tecniche costruttive
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame contenente i dati CPT
    soil_model_data : dict
        Dati del modello del suolo 3D
    max_depth : float
        Profondità massima da analizzare (m)
        
    Returns:
    --------
    FoundationAnalyzer
        Istanza dell'analizzatore con i risultati
    """
    # Crea un analizzatore
    analyzer = FoundationAnalyzer(cpt_data, soil_model_data)
    
    # Analizza la composizione del terreno
    analyzer.analyze_soil_composition(max_depth=max_depth)
    
    # Genera le raccomandazioni
    analyzer.recommend_construction_technique()
    
    return analyzer