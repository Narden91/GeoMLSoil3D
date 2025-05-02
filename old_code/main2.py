import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb
import plotly.graph_objects as go
from scipy.interpolate import LinearNDInterpolator, griddata
from glob import glob
import os
import joblib


class CPT_3D_SoilModel:
    """
    Framework for creating 3D soil models from CPT data using machine learning.
    """
    
    def __init__(self):
        """Initialize the framework components"""
        self.cpt_data = None
        self.soil_model = None
        self.scaler = None
        self.feature_columns = None
        self.soil_types = None
        self.soil_colors = None
        self.interpolator = None
    
    def load_data(self, file_pattern, x_coord_col=None, y_coord_col=None):
        """
        Load CPT data from multiple files and combine with spatial coordinates.
        
        Parameters:
        -----------
        file_pattern : str
            File pattern to match CPT files (e.g., "data/CPT_*.csv")
        x_coord_col : str, optional
            Column name containing X coordinates (Easting)
        y_coord_col : str, optional
            Column name containing Y coordinates (Northing)
        """
        print(f"Loading data from pattern: {file_pattern}")
        
        # Find all matching files
        file_paths = glob(file_pattern)
        if not file_paths:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        
        print(f"Found {len(file_paths)} CPT files")
        
        # Load and combine data
        all_data = []
        
        # Create a more interesting grid layout for artificial coordinates
        # Use a spiral pattern to ensure variety in both x and y
        def generate_spiral_coords(index, num_files):
            # Parameters for spiral
            a = 10  # Controls spacing between rings
            b = 1   # Controls how tightly wound the spiral is
            
            # Convert index to angle and radius
            theta = b * index * 2 * np.pi / num_files
            r = a * theta / (2 * np.pi)
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            return x, y
        
        for i, file_path in enumerate(file_paths):
            try:
                # Extract file name without extension
                file_name = os.path.basename(file_path).split('.')[0]
                
                # Load CSV
                df = pd.read_csv(file_path)
                
                # Add source file identifier
                df['cpt_id'] = file_name
                
                # If X and Y coordinates are not in the data, we need to add them
                if (x_coord_col is None or y_coord_col is None or 
                    x_coord_col not in df.columns or y_coord_col not in df.columns):
                    
                    print(f"No coordinates for {file_name}, generating spatial coordinates")
                    
                    # Generate coordinates using spiral pattern
                    x, y = generate_spiral_coords(i, len(file_paths))
                    
                    # Add some small variations for each point within the same CPT
                    # This is important to avoid the "same coordinate" problem
                    point_count = len(df)
                    random_state = np.random.RandomState(i)  # Use file index as seed for reproducibility
                    
                    # Generate small variations (max 1 meter in each direction)
                    x_variations = random_state.uniform(-0.5, 0.5, size=point_count)
                    y_variations = random_state.uniform(-0.5, 0.5, size=point_count)
                    
                    # Create artificial X and Y coordinates with variations
                    df['x_coord'] = x + x_variations
                    df['y_coord'] = y + y_variations
                    
                    print(f"Generated coordinates centered at ({x:.2f}, {y:.2f}) with variations")
                else:
                    # Rename to standard column names
                    df.rename(columns={x_coord_col: 'x_coord', y_coord_col: 'y_coord'}, inplace=True)
                
                all_data.append(df)
                print(f"Loaded {len(df)} records from {file_name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Combine all data
        if all_data:
            self.cpt_data = pd.concat(all_data, ignore_index=True)
            print(f"Combined dataset contains {len(self.cpt_data)} records")
            
            # Check if coordinates are all the same and warn
            if len(self.cpt_data['x_coord'].unique()) < 2:
                print("WARNING: All x coordinates are identical. 3D interpolation may fail.")
            
            if len(self.cpt_data['y_coord'].unique()) < 2:
                print("WARNING: All y coordinates are identical. 3D interpolation may fail.")
            
            # Set default feature columns if soil column exists
            if 'soil []' in self.cpt_data.columns:
                self.feature_columns = ['qc [MPa]', 'fs [MPa]', 'Rf [%]', 'u2 [MPa]']
                self.soil_types = sorted(self.cpt_data['soil []'].unique())
                
                # Create a color map for soil types
                import matplotlib.cm as cm
                import matplotlib as mpl
                
                # Usa il metodo aggiornato per ottenere una colormap (compatibile con Matplotlib ≥ 3.7)
                try:
                    # Nuovo metodo (Matplotlib ≥ 3.7)
                    cmap = mpl.colormaps['viridis'].resampled(len(self.soil_types))
                except AttributeError:
                    # Fallback per versioni precedenti
                    cmap = cm.get_cmap('viridis', len(self.soil_types))
                
                self.soil_colors = {soil_type: f'rgb({int(255*r)},{int(255*g)},{int(255*b)})' 
                            for soil_type, (r,g,b,_) in zip(self.soil_types, 
                                                            [cmap(i) for i in range(len(self.soil_types))])}
        else:
            raise ValueError("No data could be loaded")
        
        return self.cpt_data

    def explore_data(self):
        """Explore and visualize the loaded data"""
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Basic information
        print("\nDataset Overview:")
        print(f"Total records: {len(self.cpt_data)}")
        print(f"CPT locations: {self.cpt_data['cpt_id'].nunique()}")
        print(f"Columns: {', '.join(self.cpt_data.columns)}")
        
        # Check for soil classification column
        if 'soil []' in self.cpt_data.columns:
            soil_counts = self.cpt_data['soil []'].value_counts()
            print("\nSoil Type Distribution:")
            for soil_type, count in soil_counts.items():
                print(f"  Type {soil_type}: {count} records ({count/len(self.cpt_data)*100:.1f}%)")
        
        # Plot CPT locations
        plt.figure(figsize=(10, 8))
        cpt_locations = self.cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        plt.scatter(cpt_locations['x_coord'], cpt_locations['y_coord'], s=100, marker='^')
        
        for idx, row in cpt_locations.iterrows():
            plt.annotate(idx, (row['x_coord'], row['y_coord']), 
                         xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('CPT Test Locations')
        plt.grid(True)
        plt.show()
        
        # Plot example CPT profile
        if len(self.cpt_data['cpt_id'].unique()) > 0:
            example_cpt = self.cpt_data['cpt_id'].unique()[0]
            example_data = self.cpt_data[self.cpt_data['cpt_id'] == example_cpt]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 10))
            
            # Depth column is typically the first column (adjust if needed)
            depth_col = self.cpt_data.columns[0]
            
            # Plot cone resistance
            axes[0].plot(example_data['qc [MPa]'], example_data[depth_col])
            axes[0].set_ylabel('Depth (m)')
            axes[0].set_xlabel('qc [MPa]')
            axes[0].set_title('Cone Resistance')
            axes[0].grid(True)
            axes[0].invert_yaxis()  # Depth increases downward
            
            # Plot friction ratio
            axes[1].plot(example_data['Rf [%]'], example_data[depth_col])
            axes[1].set_xlabel('Rf [%]')
            axes[1].set_title('Friction Ratio')
            axes[1].grid(True)
            axes[1].invert_yaxis()
            
            # Plot soil classification if available
            if 'soil []' in self.cpt_data.columns:
                soil_data = example_data.copy()
                
                # Add some jitter to soil type for better visualization
                jitter = np.random.normal(0, 0.1, size=len(soil_data))
                soil_data['soil_jitter'] = soil_data['soil []'] + jitter
                
                axes[2].scatter(soil_data['soil_jitter'], soil_data[depth_col], 
                               c=soil_data['soil []'], cmap='viridis')
                axes[2].set_xlabel('Soil Type')
                axes[2].set_title('Soil Classification')
                axes[2].grid(True)
                axes[2].invert_yaxis()
            
            plt.tight_layout()
            plt.suptitle(f'CPT Profile: {example_cpt}', fontsize=16)
            plt.subplots_adjust(top=0.9)
            plt.show()
    
    def train_soil_classification_model(self, test_size=0.2, random_state=42, model_type='rf'):
        """
        Train a machine learning model to predict soil type from CPT parameters
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        model_type : str
            Type of model to train ('rf' for Random Forest, 'xgb' for XGBoost)
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'soil []' not in self.cpt_data.columns:
            raise ValueError("No soil classification column found in the data")
        
        print("Training soil classification model...")
        
        # Define features and target
        if self.feature_columns is None:
            # Default feature columns if not specified
            self.feature_columns = ['qc [MPa]', 'fs [MPa]', 'Rf [%]', 'u2 [MPa]']
        
        X = self.cpt_data[self.feature_columns].copy()
        y = self.cpt_data['soil []']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type.lower() == 'rf':
            # Random Forest model
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            
            base_model = RandomForestClassifier(random_state=random_state)
            
        elif model_type.lower() == 'xgb':
            # XGBoost model
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.1, 0.2]
            }
            
            base_model = xgb.XGBClassifier(random_state=random_state)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Use grid search to find best parameters
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.soil_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = self.soil_model.predict(X_test_scaled)
        
        print("\nModel Evaluation:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training accuracy: {self.soil_model.score(X_train_scaled, y_train):.4f}")
        print(f"Test accuracy: {self.soil_model.score(X_test_scaled, y_test):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(self.soil_model, 'feature_importances_'):
            importances = self.soil_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance for Soil Classification')
            plt.bar(range(X.shape[1]), importances[indices], align='center')
            plt.xticks(range(X.shape[1]), [self.feature_columns[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
        
        return self.soil_model
    
    def predict_soil_types(self):
        """
        Predict soil types for all CPT data points using the trained model
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if self.soil_model is None:
            raise ValueError("No model trained. Call train_soil_classification_model() first.")
        
        print("Predicting soil types...")
        
        # Extract features
        X = self.cpt_data[self.feature_columns].copy()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict soil types
        self.cpt_data['predicted_soil'] = self.soil_model.predict(X_scaled)
        
        # If original soil types exist, compare predictions
        if 'soil []' in self.cpt_data.columns:
            accuracy = (self.cpt_data['predicted_soil'] == self.cpt_data['soil []']).mean()
            print(f"Overall prediction accuracy: {accuracy:.4f}")
        
        return self.cpt_data['predicted_soil']
    
    def create_3d_interpolation(self, resolution=10):
        """
        Create a 3D interpolation of soil types from discrete CPT points
        
        Parameters:
        -----------
        resolution : int
            Resolution of the interpolation grid (smaller = higher resolution)
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if 'predicted_soil' not in self.cpt_data.columns:
            raise ValueError("No soil predictions. Call predict_soil_types() first.")
        
        print("Creating 3D soil interpolation...")
        
        # Get the soil column to use
        soil_col = 'predicted_soil'
        
        # Extract coordinates and soil types
        points = self.cpt_data[['x_coord', 'y_coord', self.cpt_data.columns[0]]].values  # Using first column as depth
        values = self.cpt_data[soil_col].values
        
        # Check if all x coordinates are the same
        x_coords = self.cpt_data['x_coord'].unique()
        y_coords = self.cpt_data['y_coord'].unique()
        
        print(f"Unique x coordinates: {len(x_coords)}")
        print(f"Unique y coordinates: {len(y_coords)}")
        
        # If we have a degenerate case (all points have same x or y), we need to add some jitter
        if len(x_coords) < 2 or len(y_coords) < 2:
            print("WARNING: Not enough unique coordinates for interpolation. Adding artificial jitter...")
            
            # Create a copy to avoid modifying original data
            jittered_points = points.copy()
            
            # Add small random values to make coordinates unique
            # Scale jitter based on the depth range to maintain reasonable proportions
            depth_range = self.cpt_data[self.cpt_data.columns[0]].max() - self.cpt_data[self.cpt_data.columns[0]].min()
            jitter_scale = depth_range * 0.01  # 1% of depth range
            
            if len(x_coords) < 2:
                jittered_points[:, 0] += np.random.uniform(-jitter_scale, jitter_scale, size=jittered_points.shape[0])
                print(f"Added jitter to x coordinates with scale {jitter_scale}")
                
            if len(y_coords) < 2:
                jittered_points[:, 1] += np.random.uniform(-jitter_scale, jitter_scale, size=jittered_points.shape[0])
                print(f"Added jitter to y coordinates with scale {jitter_scale}")
            
            # Replace original points with jittered points
            points = jittered_points
        
        # Try to create the interpolator - first with LinearNDInterpolator, but fallback to griddata if it fails
        try:
            print("Attempting to create LinearNDInterpolator...")
            self.interpolator = LinearNDInterpolator(points, values, fill_value=-1)
            interpolator_type = "LinearNDInterpolator"
        except Exception as e:
            print(f"LinearNDInterpolator failed: {e}")
            print("Falling back to griddata interpolator...")
            
            # griddata is more robust but slower
            def griddata_interpolator(pts):
                return griddata(points, values, pts, method='linear', fill_value=-1)
            
            self.interpolator = griddata_interpolator
            interpolator_type = "griddata"
        
        print(f"Using {interpolator_type} for interpolation")
        
        # Create a grid for interpolation
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        # Add some margin
        margin = 5  # meters
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Ensure we have a reasonable range even with jittered values
        if x_max - x_min < resolution:
            center = (x_max + x_min) / 2
            x_min = center - resolution
            x_max = center + resolution
        
        if y_max - y_min < resolution:
            center = (y_max + y_min) / 2
            y_min = center - resolution
            y_max = center + resolution
        
        # Create grid
        x_range = np.arange(x_min, x_max, resolution)
        y_range = np.arange(y_min, y_max, resolution)
        z_range = np.arange(z_min, z_max, resolution/2)  # Higher vertical resolution
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        
        # Flatten the grid
        grid_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        
        # Interpolate soil types
        print(f"Interpolating {len(grid_points)} points...")
        
        # Handle different interpolator types
        if interpolator_type == "LinearNDInterpolator":
            grid_values = self.interpolator(grid_points)
        else:
            grid_values = self.interpolator(grid_points)
        
        # Reshape back to grid
        grid_values = grid_values.reshape(X.shape)
        
        return {
            'X': X,
            'Y': Y, 
            'Z': Z,
            'values': grid_values
        }
    
    def create_3d_interpolation_alternative(self, resolution=10):
        """
        Create a 3D interpolation of soil types using nearest neighbor method
        which is more robust than LinearNDInterpolator for problematic datasets
        
        Parameters:
        -----------
        resolution : int
            Resolution of the interpolation grid (smaller = higher resolution)
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if 'predicted_soil' not in self.cpt_data.columns:
            raise ValueError("No soil predictions. Call predict_soil_types() first.")
        
        print("Creating 3D soil interpolation using nearest neighbor method...")
        
        # Get the soil column to use
        soil_col = 'predicted_soil'
        
        # Extract coordinates and soil types
        depth_col = self.cpt_data.columns[0]
        coords_df = self.cpt_data[['x_coord', 'y_coord', depth_col]].copy()
        values = self.cpt_data[soil_col].values
        
        # Check coordinate diversity
        x_unique = len(coords_df['x_coord'].unique())
        y_unique = len(coords_df['y_coord'].unique())
        print(f"Dataset has {x_unique} unique x-coordinates and {y_unique} unique y-coordinates")
        
        # Create grid for interpolation
        x_min, x_max = coords_df['x_coord'].min(), coords_df['x_coord'].max()
        y_min, y_max = coords_df['y_coord'].min(), coords_df['y_coord'].max()
        z_min, z_max = coords_df[depth_col].min(), coords_df[depth_col].max()
        
        # Ensure we have some range in the coordinates
        if x_max == x_min:
            print("All x-coordinates are identical. Adding artificial range...")
            range_value = max(10, z_max - z_min)  # Use depth range as a reference
            x_min -= range_value / 2
            x_max += range_value / 2
        
        if y_max == y_min:
            print("All y-coordinates are identical. Adding artificial range...")
            range_value = max(10, z_max - z_min)  # Use depth range as a reference
            y_min -= range_value / 2
            y_max += range_value / 2
        
        # Add some margin
        margin = 5  # meters
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Create grid with specified resolution
        x_range = np.linspace(x_min, x_max, int((x_max - x_min) / resolution) + 1)
        y_range = np.linspace(y_min, y_max, int((y_max - y_min) / resolution) + 1)
        z_range = np.linspace(z_min, z_max, int((z_max - z_min) / (resolution/2)) + 1)  # Higher vertical resolution
        
        print(f"Created grid with dimensions: {len(x_range)}x{len(y_range)}x{len(z_range)}")
        
        # Create meshgrid
        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        
        # Prepare for nearest neighbor interpolation
        from scipy.spatial import cKDTree
        
        # Create KD-tree from the source points
        source_points = coords_df[['x_coord', 'y_coord', depth_col]].values
        tree = cKDTree(source_points)
        
        # Prepare target points (flatten the grid)
        target_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        
        print(f"Finding nearest neighbors for {len(target_points)} grid points...")
        
        # Find nearest neighbors
        distances, indices = tree.query(target_points, k=1)
        
        # Get soil values for grid points
        grid_values = values[indices]
        
        # Reshape back to grid
        grid_values = grid_values.reshape(X.shape)
        
        # Create a "fake" interpolator function for compatibility with existing code
        def nn_interpolator(points):
            """Nearest neighbor interpolator function"""
            _, idx = tree.query(points, k=1)
            return values[idx]
        
        self.interpolator = nn_interpolator
        
        return {
            'X': X,
            'Y': Y, 
            'Z': Z,
            'values': grid_values
        }
    
    def visualize_3d_model(self, interpolation_data=None, interactive=True):
        """
        Visualize the 3D soil model
        
        Parameters:
        -----------
        interpolation_data : dict, optional
            Output from create_3d_interpolation()
        interactive : bool
            Whether to create an interactive Plotly visualization
        """
        if interpolation_data is None:
            interpolation_data = self.create_3d_interpolation()
        
        X = interpolation_data['X']
        Y = interpolation_data['Y']
        Z = interpolation_data['Z']
        values = interpolation_data['values']
        
        print("Creating 3D visualization...")
        
        if interactive:
            # Create interactive 3D visualization with Plotly
            fig = go.Figure()
            
            # Create colormap
            if self.soil_colors is None:
                # Default colormap
                colorscale = 'Viridis'
            else:
                # Custom colormap from soil types
                colorscale = []
                soil_types = sorted(self.soil_colors.keys())
                for i, soil_type in enumerate(soil_types):
                    normalized_val = i / (len(soil_types) - 1)
                    colorscale.append([normalized_val, self.soil_colors[soil_type]])
            
            # Add a volume trace
            fig.add_trace(go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                isomin=min(self.soil_types) if self.soil_types else 0,
                isomax=max(self.soil_types) if self.soil_types else 10,
                opacity=0.6,
                surface_count=15,  # Controls the level of detail
                colorscale=colorscale,
                colorbar=dict(
                    title='Soil Type',
                    tickvals=self.soil_types if self.soil_types else None
                )
            ))
            
            # Add CPT locations for reference
            cpt_locations = self.cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
            fig.add_trace(go.Scatter3d(
                x=cpt_locations['x_coord'],
                y=cpt_locations['y_coord'],
                z=[Z.min() for _ in range(len(cpt_locations))],  # Place at surface
                mode='markers',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='circle'
                ),
                text=cpt_locations.index,
                name='CPT Locations'
            ))
            
            # Update layout
            fig.update_layout(
                title='3D Soil Composition Model',
                scene=dict(
                    xaxis_title='X Coordinate (m)',
                    yaxis_title='Y Coordinate (m)',
                    zaxis_title='Depth (m)',
                    aspectratio=dict(x=1, y=1, z=0.5),  # Vertical exaggeration
                    zaxis=dict(autorange='reversed')  # Invert Z axis for depth
                ),
                width=900,
                height=800,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            # Add sliders for cross-sections
            # (This would be more complex in a full implementation)
            
            fig.show()
            return fig
        
        else:
            # Create static 3D visualization with matplotlib
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # We'll use a scatter plot with points colored by soil type
            # Sampling the grid to avoid too many points
            sample_step = 3
            
            # Create scatter plot of soil types
            scatter = ax.scatter(
                X[::sample_step, ::sample_step, ::sample_step].flatten(),
                Y[::sample_step, ::sample_step, ::sample_step].flatten(),
                Z[::sample_step, ::sample_step, ::sample_step].flatten(),
                c=values[::sample_step, ::sample_step, ::sample_step].flatten(),
                cmap='viridis',
                alpha=0.3,
                marker='o',
                s=5
            )
            
            # Add CPT locations
            cpt_locations = self.cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
            ax.scatter(
                cpt_locations['x_coord'],
                cpt_locations['y_coord'],
                [Z.min() for _ in range(len(cpt_locations))],
                color='black',
                marker='^',
                s=100,
                label='CPT Locations'
            )
            
            # Set labels and title
            ax.set_xlabel('X Coordinate (m)')
            ax.set_ylabel('Y Coordinate (m)')
            ax.set_zlabel('Depth (m)')
            ax.set_title('3D Soil Composition Model')
            
            # Invert z-axis for depth
            ax.invert_zaxis()
            
            # Add a color bar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Soil Type')
            
            plt.tight_layout()
            plt.show()
            
            return fig
    
    def save_model(self, filename='cpt_soil_model.pkl'):
        """
        Save the trained model and related components
        
        Parameters:
        -----------
        filename : str
            Filename to save the model
        """
        if self.soil_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'soil_model': self.soil_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'soil_types': self.soil_types,
            'soil_colors': self.soil_colors
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='cpt_soil_model.pkl'):
        """
        Load a previously saved model
        
        Parameters:
        -----------
        filename : str
            Filename to load the model from
        """
        model_data = joblib.load(filename)
        
        self.soil_model = model_data['soil_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.soil_types = model_data['soil_types']
        self.soil_colors = model_data['soil_colors']
        
        print(f"Model loaded from {filename}")


if __name__ == "__main__":
    from debug_cpt import debug_cpt_files
    
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