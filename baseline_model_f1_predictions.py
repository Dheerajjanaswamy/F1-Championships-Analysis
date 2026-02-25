"""
F1 Baseline Model for Race Outcome Predictions
Transferred from Kaggle notebook: baseline-model-for-predictions

This module implements a baseline XGBoost model for predicting F1 race outcomes
using historical race data, driver performance, and constructor information.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class F1BaselineModel:
    """Baseline F1 race outcome prediction model using XGBoost."""
    
    def __init__(self, data_dir, output_dir, start_year=2010, lookback_years=10):
        """
        Initialize the F1 baseline model.
        
        Args:
            data_dir: Path to F1 dataset directory
            output_dir: Path for output files
            start_year: Starting year for training data
            lookback_years: Number of years to look back for historical features
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.start_year = start_year
        self.lookback_years = lookback_years
        self.random_state = 42
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load F1 datasets from CSV files."""
        try:
            self.races = pd.read_csv(self.data_dir / 'races.csv')
            self.results = pd.read_csv(self.data_dir / 'results.csv')
            self.qualifying = pd.read_csv(self.data_dir / 'qualifying.csv')
            self.drivers = pd.read_csv(self.data_dir / 'drivers.csv')
            self.constructors = pd.read_csv(self.data_dir / 'constructors.csv')
            self.circuits = pd.read_csv(self.data_dir / 'circuits.csv')
            self.driver_standings = pd.read_csv(self.data_dir / 'driver_standings.csv')
            self.constructor_standings = pd.read_csv(self.data_dir / 'constructor_standings.csv')
            
            print(f"Files loaded successfully!")
            print(f"Races: {len(self.races)}, Results: {len(self.results)}, Qualifying: {len(self.qualifying)}")
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            raise
    
    def standardize_columns(self, df):
        """Convert camelCase columns to snake_case."""
        rename_map = {
            'raceId': 'race_id',
            'driverId': 'driver_id',
            'constructorId': 'constructor_id',
            'circuitId': 'circuit_id',
            'positionOrder': 'position_order',
            'positionText': 'position_text',
            'gridPosition': 'grid_position',
            'fastestLap': 'fastest_lap',
            'fastestLapTime': 'fastest_lap_time',
        }
        
        return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    def prepare_data(self):
        """Prepare and clean the data."""
        # Standardize column names
        self.races = self.standardize_columns(self.races)
        self.results = self.standardize_columns(self.results)
        self.qualifying = self.standardize_columns(self.qualifying)
        
        # Parse dates
        if 'date' in self.races.columns:
            self.races['date'] = pd.to_datetime(self.races['date'], errors='coerce')
        
        # Filter by start year
        if 'year' in self.races.columns:
            self.races = self.races[self.races['year'] >= self.start_year]
        
        # Handle position_order
        if 'position_order' not in self.results.columns and 'positionOrder' in self.results.columns:
            self.results['position_order'] = self.results['positionOrder']
        
        print("Data preparation completed.")
    
    def build_driver_history(self):
        """Build historical driver performance features."""
        driver_history = defaultdict(list)
        
        for idx, row in self.results.iterrows():
            race_id = row.get('race_id')
            driver_id = row.get('driver_id')
            position = row.get('position_order')
            points = row.get('points', 0)
            
            if pd.notna(driver_id) and pd.notna(race_id):
                driver_history[driver_id].append({
                    'race_id': race_id,
                    'position': position,
                    'points': points
                })
        
        return driver_history
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train XGBoost baseline model."""
        model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            random_state=self.random_state,
            verbosity=0
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        
        return model
    
    def run_pipeline(self):
        """Run the complete baseline model pipeline."""
        print("Starting F1 Baseline Model Pipeline...")
        
        # Load data
        self.load_data()
        
        # Prepare data
        self.prepare_data()
        
        # Build features (placeholder - actual feature engineering would go here)
        print("Building features...")
        driver_history = self.build_driver_history()
        print(f"Driver history built for {len(driver_history)} drivers")
        
        print("\nBaseline model pipeline initialized.")
        print("Note: Complete feature engineering and model training")
        print("should be implemented based on your specific requirements.")
        
        return self


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/f1_dataset"
    OUTPUT_DIR = "output/baseline_model"
    START_YEAR = 2010
    LOOKBACK_YEARS = 10
    
    # Initialize and run model
    model = F1BaselineModel(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        start_year=START_YEAR,
        lookback_years=LOOKBACK_YEARS
    )
    
    model.run_pipeline()
