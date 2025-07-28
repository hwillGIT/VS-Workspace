"""
Comprehensive tests for the ML Ensemble Agent.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.ml_ensemble.ml_ensemble_agent import MLEnsembleAgent
from core.base.exceptions import ModelError, ValidationError


@pytest.fixture
def ml_agent():
    """Create an MLEnsembleAgent instance for testing."""
    return MLEnsembleAgent()


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    features = pd.DataFrame({
        'price_change': np.random.normal(0, 0.02, n_samples),
        'volume_ratio': np.random.lognormal(0, 0.3, n_samples),
        'volatility': np.random.gamma(2, 0.01, n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.normal(0, 0.5, n_samples),
        'sma_ratio': np.random.normal(1, 0.05, n_samples)
    })
    
    # Create target (next day return)
    target = (
        0.1 * features['price_change'] +
        0.05 * (features['rsi'] - 50) / 50 +
        0.03 * features['macd'] +
        np.random.normal(0, 0.01, n_samples)
    )
    
    return features, target


@pytest.fixture
def sample_prediction_data():
    """Create sample data for predictions."""
    np.random.seed(123)
    n_samples = 100
    
    return pd.DataFrame({
        'price_change': np.random.normal(0, 0.02, n_samples),
        'volume_ratio': np.random.lognormal(0, 0.3, n_samples),
        'volatility': np.random.gamma(2, 0.01, n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.normal(0, 0.5, n_samples),
        'sma_ratio': np.random.normal(1, 0.05, n_samples)
    })


@pytest.fixture
def sample_inputs():
    """Sample inputs for ML ensemble."""
    return {
        "symbols": ["AAPL", "GOOGL"],
        "features": ["technical", "fundamental", "sentiment"],
        "prediction_horizon": 5,
        "model_types": ["rf", "xgb", "lstm"],
        "retrain": False
    }


class TestMLEnsembleAgent:
    """Test cases for MLEnsembleAgent."""
    
    @pytest.mark.unit
    def test_agent_initialization(self, ml_agent):
        """Test agent initialization."""
        assert ml_agent.name == "MLEnsembleAgent"
        assert ml_agent.config_section == "ml_ensemble"
        assert hasattr(ml_agent, 'models')
        assert hasattr(ml_agent, 'scaler')
    
    @pytest.mark.unit
    def test_feature_engineering(self, ml_agent, sample_training_data):
        """Test feature engineering."""
        features, _ = sample_training_data
        
        engineered = ml_agent._engineer_features(features)
        
        assert isinstance(engineered, pd.DataFrame)
        assert len(engineered) == len(features)
        assert engineered.shape[1] >= features.shape[1]  # Should add features
    
    @pytest.mark.unit
    def test_feature_selection(self, ml_agent, sample_training_data):
        """Test feature selection."""
        features, target = sample_training_data
        
        selected_features = ml_agent._select_features(features, target, max_features=4)
        
        assert isinstance(selected_features, list)
        assert len(selected_features) <= 4
        assert all(feat in features.columns for feat in selected_features)
    
    @pytest.mark.unit
    def test_model_training_random_forest(self, ml_agent, sample_training_data):
        """Test Random Forest model training."""
        features, target = sample_training_data
        
        model = ml_agent._train_random_forest(features, target)
        
        assert hasattr(model, 'predict')
        assert hasattr(model, 'feature_importances_')
        
        # Test prediction
        predictions = model.predict(features.iloc[:10])
        assert len(predictions) == 10
    
    @pytest.mark.unit
    def test_model_training_linear_regression(self, ml_agent, sample_training_data):
        """Test Linear Regression model training."""
        features, target = sample_training_data
        
        model = ml_agent._train_linear_regression(features, target)
        
        assert hasattr(model, 'predict')
        assert hasattr(model, 'coef_')
        
        # Test prediction
        predictions = model.predict(features.iloc[:10])
        assert len(predictions) == 10
    
    @pytest.mark.unit
    def test_ensemble_prediction(self, ml_agent, sample_training_data, sample_prediction_data):
        """Test ensemble prediction."""
        features, target = sample_training_data
        
        # Train multiple models
        rf_model = ml_agent._train_random_forest(features, target)
        lr_model = ml_agent._train_linear_regression(features, target)
        
        models = {'rf': rf_model, 'lr': lr_model}
        weights = {'rf': 0.6, 'lr': 0.4}
        
        ensemble_pred = ml_agent._ensemble_predict(models, weights, sample_prediction_data)
        
        assert len(ensemble_pred) == len(sample_prediction_data)
        assert isinstance(ensemble_pred, np.ndarray)
    
    @pytest.mark.unit
    def test_model_validation(self, ml_agent, sample_training_data):
        """Test model validation."""
        features, target = sample_training_data
        
        # Split data
        split_idx = int(0.8 * len(features))
        train_X, val_X = features[:split_idx], features[split_idx:]
        train_y, val_y = target[:split_idx], target[split_idx:]
        
        model = ml_agent._train_random_forest(train_X, train_y)
        
        metrics = ml_agent._validate_model(model, val_X, val_y)
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    @pytest.mark.unit
    def test_hyperparameter_optimization(self, ml_agent, sample_training_data):
        """Test hyperparameter optimization for Random Forest."""
        features, target = sample_training_data
        
        # Use small subset for faster testing
        features_small = features.iloc[:200]
        target_small = target.iloc[:200]
        
        best_params = ml_agent._optimize_hyperparameters(
            'rf', features_small, target_small, n_trials=5
        )
        
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
    
    @pytest.mark.unit
    def test_feature_importance_analysis(self, ml_agent, sample_training_data):
        """Test feature importance analysis."""
        features, target = sample_training_data
        
        model = ml_agent._train_random_forest(features, target)
        
        importance = ml_agent._analyze_feature_importance(model, features.columns)
        
        assert isinstance(importance, dict)
        assert len(importance) == len(features.columns)
        assert all(isinstance(v, float) for v in importance.values())
    
    @pytest.mark.unit
    def test_prediction_intervals(self, ml_agent, sample_training_data, sample_prediction_data):
        """Test prediction interval calculation."""
        features, target = sample_training_data
        
        model = ml_agent._train_random_forest(features, target)
        
        predictions, intervals = ml_agent._calculate_prediction_intervals(
            model, sample_prediction_data, confidence=0.95
        )
        
        assert len(predictions) == len(sample_prediction_data)
        assert 'lower' in intervals
        assert 'upper' in intervals
        assert len(intervals['lower']) == len(sample_prediction_data)
        assert all(intervals['lower'] <= intervals['upper'])
    
    @pytest.mark.unit
    def test_model_drift_detection(self, ml_agent, sample_training_data):
        """Test model drift detection."""
        features, target = sample_training_data
        
        # Create slightly different data to simulate drift
        np.random.seed(999)
        drifted_features = features + np.random.normal(0, 0.1, features.shape)
        
        drift_score = ml_agent._detect_model_drift(features, drifted_features)
        
        assert isinstance(drift_score, float)
        assert drift_score >= 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_with_mock_data(self, ml_agent, sample_inputs):
        """Test processing with mocked data."""
        # Mock data fetching and model training
        with patch.object(ml_agent, '_fetch_training_data') as mock_fetch:
            mock_fetch.return_value = sample_training_data
            
            with patch.object(ml_agent, '_load_pretrained_models') as mock_load:
                mock_models = {
                    'rf': RandomForestRegressor(n_estimators=10, random_state=42),
                    'lr': LinearRegression()
                }
                mock_load.return_value = mock_models
                
                result = await ml_agent.process(sample_inputs)
                
                assert isinstance(result, dict)
                assert 'predictions' in result
                assert 'confidence' in result
                assert 'model_performance' in result
    
    @pytest.mark.unit
    def test_input_validation(self, ml_agent):
        """Test input validation."""
        # Test missing required fields
        invalid_inputs = {"symbols": ["AAPL"]}  # Missing prediction_horizon
        
        with pytest.raises(ValidationError):
            ml_agent._validate_inputs(invalid_inputs)
        
        # Test invalid prediction horizon
        invalid_horizon = {
            "symbols": ["AAPL"],
            "prediction_horizon": -1  # Invalid negative horizon
        }
        
        with pytest.raises(ValidationError):
            ml_agent._validate_inputs(invalid_horizon)
    
    @pytest.mark.unit
    def test_data_preprocessing(self, ml_agent, sample_training_data):
        """Test data preprocessing."""
        features, target = sample_training_data
        
        # Add some NaN values
        features_with_na = features.copy()
        features_with_na.iloc[0, 0] = np.nan
        features_with_na.iloc[10, 2] = np.nan
        
        cleaned_features, cleaned_target = ml_agent._preprocess_data(features_with_na, target)
        
        assert not cleaned_features.isnull().any().any()
        assert len(cleaned_features) == len(cleaned_target)
        assert len(cleaned_features) <= len(features)
    
    @pytest.mark.unit
    def test_model_persistence(self, ml_agent, sample_training_data):
        """Test model saving and loading."""
        features, target = sample_training_data
        
        model = ml_agent._train_random_forest(features.iloc[:100], target.iloc[:100])
        
        # Test model serialization (mock file operations)
        with patch('joblib.dump') as mock_dump:
            with patch('joblib.load') as mock_load:
                mock_load.return_value = model
                
                ml_agent._save_model(model, 'test_model.pkl')
                loaded_model = ml_agent._load_model('test_model.pkl')
                
                mock_dump.assert_called_once()
                mock_load.assert_called_once()
    
    @pytest.mark.unit
    def test_ensemble_weights_optimization(self, ml_agent, sample_training_data):
        """Test ensemble weights optimization."""
        features, target = sample_training_data
        
        # Train multiple models
        models = {
            'rf': ml_agent._train_random_forest(features.iloc[:500], target.iloc[:500]),
            'lr': ml_agent._train_linear_regression(features.iloc[:500], target.iloc[:500])
        }
        
        # Use validation data
        val_features = features.iloc[500:600]
        val_target = target.iloc[500:600]
        
        optimal_weights = ml_agent._optimize_ensemble_weights(models, val_features, val_target)
        
        assert isinstance(optimal_weights, dict)
        assert set(optimal_weights.keys()) == set(models.keys())
        assert abs(sum(optimal_weights.values()) - 1.0) < 1e-6  # Weights should sum to 1
    
    @pytest.mark.unit
    def test_cross_validation(self, ml_agent, sample_training_data):
        """Test cross-validation."""
        features, target = sample_training_data
        
        # Use smaller dataset for faster testing
        features_small = features.iloc[:200]
        target_small = target.iloc[:200]
        
        cv_scores = ml_agent._cross_validate_model(
            'rf', features_small, target_small, cv_folds=3
        )
        
        assert isinstance(cv_scores, dict)
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        assert 'scores' in cv_scores
        assert len(cv_scores['scores']) == 3
    
    @pytest.mark.unit
    def test_prediction_confidence(self, ml_agent, sample_training_data, sample_prediction_data):
        """Test prediction confidence calculation."""
        features, target = sample_training_data
        
        models = {
            'rf': ml_agent._train_random_forest(features.iloc[:500], target.iloc[:500]),
            'lr': ml_agent._train_linear_regression(features.iloc[:500], target.iloc[:500])
        }
        
        predictions = ml_agent._ensemble_predict(
            models, {'rf': 0.6, 'lr': 0.4}, sample_prediction_data.iloc[:10]
        )
        
        confidence = ml_agent._calculate_prediction_confidence(
            models, sample_prediction_data.iloc[:10], predictions
        )
        
        assert isinstance(confidence, np.ndarray)
        assert len(confidence) == 10
        assert all(0 <= c <= 1 for c in confidence)
    
    @pytest.mark.integration
    async def test_online_learning(self, ml_agent, sample_training_data):
        """Test online learning capability."""
        features, target = sample_training_data
        
        # Initial training
        initial_features = features.iloc[:500]
        initial_target = target.iloc[:500]
        
        model = ml_agent._train_random_forest(initial_features, initial_target)
        initial_score = model.score(features.iloc[500:600], target.iloc[500:600])
        
        # Simulate new data arrival
        new_features = features.iloc[600:700]
        new_target = target.iloc[600:700]
        
        updated_model = ml_agent._update_model_online(model, new_features, new_target)
        
        # Verify model can still make predictions
        predictions = updated_model.predict(features.iloc[700:710])
        assert len(predictions) == 10
    
    @pytest.mark.slow
    async def test_end_to_end_training_pipeline(self, ml_agent):
        """Test complete training pipeline."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 2000
        
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.uniform(-1, 1, n_samples),
            'feature_3': np.random.gamma(2, 1, n_samples),
            'feature_4': np.random.beta(2, 5, n_samples)
        })
        
        target = (
            0.5 * features['feature_1'] +
            0.3 * features['feature_2'] +
            0.2 * features['feature_3'] +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Run complete pipeline
        with patch.object(ml_agent, '_fetch_training_data', return_value=(features, target)):
            pipeline_result = await ml_agent._run_training_pipeline({
                "model_types": ["rf", "lr"],
                "validation_split": 0.2,
                "hyperparameter_tuning": True
            })
            
            assert 'trained_models' in pipeline_result
            assert 'validation_scores' in pipeline_result
            assert 'feature_importance' in pipeline_result