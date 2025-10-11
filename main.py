import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from io import StringIO, BytesIO
import chardet
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, GlobalMaxPooling1D, 
                                   LSTM, GRU, Bidirectional, BatchNormalization,
                                   Input, MultiHeadAttention, LayerNormalization,
                                   GlobalAveragePooling1D, Add, Flatten, Activation)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor)
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                           mean_squared_error, mean_absolute_error, r2_score, classification_report, 
                           confusion_matrix, precision_recall_curve, roc_curve, mean_absolute_percentage_error)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, 
                                 OneHotEncoder, PowerTransformer, QuantileTransformer)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression, mutual_info_classif, 
                                     mutual_info_regression, RFE, RFECV, SelectFromModel)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from scipy.stats import normaltest, shapiro, jarque_bera, anderson, pearsonr, spearmanr
import time
import pickle
import json
import warnings
import joblib
from datetime import datetime
import os
import sys
from pathlib import Path
import base64

warnings.filterwarnings('ignore')

# 🔧 Environment settings to prevent multiprocessing issues
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set page configuration
st.set_page_config(
    page_title="Expert AutoML & Deep Learning Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 2rem;
        color: #4ECDC4;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        border: none;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .warning-box {
        padding: 1.5rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        border: none;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

class ExpertMLPipeline:
    """Expert-level Machine Learning Pipeline with Advanced Features"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.hyperparameters = {}
        
    def get_all_models(self, task_type='classification'):
        """Get all available models for the given task type"""
        if task_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
                'XGBoost': XGBClassifier(random_state=42, n_jobs=1, eval_metric='logloss'),
                'LightGBM': LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
                'Logistic Regression': LogisticRegression(random_state=42, n_jobs=1, max_iter=1000),
                'SVM': SVC(probability=True, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=1),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'Gaussian Naive Bayes': GaussianNB(),
                'MLP Classifier': MLPClassifier(random_state=42, max_iter=1000)
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
                'XGBoost': XGBRegressor(random_state=42, n_jobs=1),
                'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),
                'Linear Regression': LinearRegression(n_jobs=1),
                'Ridge Regression': Ridge(random_state=42),
                'Lasso Regression': Lasso(random_state=42),
                'SVR': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'K-Nearest Neighbors': KNeighborsRegressor(n_jobs=1),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42),
                'MLP Regressor': MLPRegressor(random_state=42, max_iter=1000)
            }
    
    def get_best_models_auto(self, evaluation_results, task_type='classification', top_k=3):
        """Automatically select best models based on evaluation results"""
        if not evaluation_results:
            return self.get_default_best_models(task_type)
        
        if task_type == 'classification':
            # Sort by accuracy and select top k models
            sorted_models = sorted(evaluation_results.items(), 
                                 key=lambda x: x[1].get('accuracy', 0), 
                                 reverse=True)
        else:
            # Sort by R² score and select top k models
            sorted_models = sorted(evaluation_results.items(), 
                                 key=lambda x: x[1].get('r2_score', -float('inf')), 
                                 reverse=True)
        
        best_models = {name: evaluation_results[name]['model'] for name, _ in sorted_models[:top_k]}
        return best_models
    
    def get_default_best_models(self, task_type='classification'):
        """Get default best models if no evaluation results available"""
        if task_type == 'classification':
            return {
                'XGBoost': XGBClassifier(random_state=42, n_jobs=1, eval_metric='logloss'),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
                'LightGBM': LGBMClassifier(random_state=42, verbose=-1, n_jobs=1)
            }
        else:
            return {
                'XGBoost': XGBRegressor(random_state=42, n_jobs=1),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
                'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=1)
            }

class AdvancedDeepLearningModels:
    """Advanced Deep Learning Models with Modern Architectures"""
    
    def __init__(self):
        self.models = {}
    
    def get_all_dl_models(self, input_shape, task_type='classification'):
        """Get all available deep learning models"""
        output_activation = 'sigmoid' if task_type == 'classification' else 'linear'
        
        return {
            'Simple Neural Network': self.create_simple_nn(input_shape, output_activation=output_activation),
            'Residual MLP': self.create_residual_mlp(input_shape, output_activation=output_activation),
            'Transformer Model': self.create_tabular_transformer(input_shape, output_activation=output_activation),
            'Hybrid CNN-LSTM': self.create_hybrid_cnn_lstm(input_shape, output_activation=output_activation)
        }
    
    def get_best_dl_models_auto(self, evaluation_results, task_type='classification', top_k=2):
        """Automatically select best DL models based on evaluation results"""
        if not evaluation_results:
            return self.get_default_best_dl_models(task_type)
        
        if task_type == 'classification':
            # Sort by accuracy and select top k models
            sorted_models = sorted(evaluation_results.items(), 
                                 key=lambda x: x[1].get('accuracy', 0), 
                                 reverse=True)
        else:
            # Sort by R² score and select top k models
            sorted_models = sorted(evaluation_results.items(), 
                                 key=lambda x: x[1].get('r2_score', -float('inf')), 
                                 reverse=True)
        
        best_models = {name: evaluation_results[name]['model'] for name, _ in sorted_models[:top_k]}
        return best_models
    
    def get_default_best_dl_models(self, task_type='classification'):
        """Get default best DL models"""
        return {
            'Simple Neural Network': None,
            'Residual MLP': None
        }
    
    def create_simple_nn(self, input_shape, output_activation='sigmoid'):
        """Create a simple neural network"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1 if output_activation in ['sigmoid', 'linear'] else 2, activation=output_activation)
        ])
        
        if output_activation == 'sigmoid':
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        elif output_activation == 'softmax':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
            
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=metrics)
        return model

    def create_transformer_block(self, input_tensor, head_size, num_heads, ff_dim, dropout=0):
        """Create a transformer block for tabular data"""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(input_tensor, input_tensor)
        
        # Add & Norm
        x = Add()([input_tensor, attention_output])
        x = LayerNormalization()(x)
        
        # Feed Forward
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dense(input_tensor.shape[-1])(ff_output)
        
        # Add & Norm
        x = Add()([x, ff_output])
        x = LayerNormalization()(x)
        
        return x
    
    def create_tabular_transformer(self, input_shape, num_transformer_blocks=2, 
                                 head_size=256, num_heads=4, ff_dim=512, 
                                 mlp_units=[128, 64], dropout=0.1, output_activation='sigmoid'):
        """Create a transformer model for tabular data"""
        
        inputs = Input(shape=input_shape)
        
        # Initial projection
        x = Dense(head_size)(inputs)
        x = tf.keras.layers.Reshape((1, head_size))(x)  # Add sequence dimension
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = self.create_transformer_block(x, head_size, num_heads, ff_dim, dropout)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # MLP head
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(dropout)(x)
        
        # Output layer
        if output_activation == 'sigmoid':
            outputs = Dense(1, activation="sigmoid")(x)
            loss = 'binary_crossentropy'
        elif output_activation == 'softmax':
            outputs = Dense(2, activation="softmax")(x)  # Assuming binary classification
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1)(x)
            loss = 'mse'
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=loss,
            metrics=['accuracy'] if output_activation in ['sigmoid', 'softmax'] else ['mae']
        )
        
        return model
    
    def create_residual_mlp(self, input_shape, hidden_units=[512, 256, 128, 64], 
                          dropout_rate=0.3, output_activation='sigmoid'):
        """Create MLP with residual connections"""
        
        inputs = Input(shape=input_shape)
        x = Dense(hidden_units[0], activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Residual blocks
        for units in hidden_units[1:]:
            # Residual connection
            residual = x
            
            # Main path
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            
            # Adjust residual if dimensions don't match
            if residual.shape[-1] != units:
                residual = Dense(units)(residual)
            
            # Add residual
            x = Add()([x, residual])
            x = Activation('relu')(x)
            x = Dropout(dropout_rate)(x)
        
        # Output layer
        if output_activation == 'sigmoid':
            outputs = Dense(1, activation="sigmoid")(x)
            loss = 'binary_crossentropy'
        elif output_activation == 'softmax':
            outputs = Dense(2, activation="softmax")(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1)(x)
            loss = 'mse'
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=loss,
            metrics=['accuracy'] if output_activation in ['sigmoid', 'softmax'] else ['mae']
        )
        
        return model
    
    def create_hybrid_cnn_lstm(self, input_shape, cnn_filters=[64, 128], 
                             lstm_units=[64, 32], dense_units=[64], 
                             output_activation='sigmoid'):
        """Create hybrid CNN-LSTM model for tabular data"""
        
        inputs = Input(shape=input_shape)
        
        # Reshape for CNN (add channel dimension)
        x = tf.keras.layers.Reshape((input_shape[0], 1))(inputs)
        
        # CNN layers
        for filters in cnn_filters:
            x = Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = LSTM(units, return_sequences=return_sequences, dropout=0.2)(x)
        
        # Dense layers
        for units in dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.3)(x)
        
        # Output layer
        if output_activation == 'sigmoid':
            outputs = Dense(1, activation="sigmoid")(x)
            loss = 'binary_crossentropy'
        elif output_activation == 'softmax':
            outputs = Dense(2, activation="softmax")(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1)(x)
            loss = 'mse'
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=loss,
            metrics=['accuracy'] if output_activation in ['sigmoid', 'softmax'] else ['mae']
        )
        
        return model

class ExpertModelTrainer:
    """Expert-level model training with advanced techniques"""
    
    def __init__(self):
        self.ml_pipeline = ExpertMLPipeline()
        self.dl_models = AdvancedDeepLearningModels()
        self.training_history = {}
    
    def create_advanced_callbacks(self):
        """Create advanced training callbacks"""
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def quick_evaluate_models(self, X_train, y_train, X_test, y_test, task_type='classification'):
        """Quickly evaluate all models to select the best ones"""
        st.subheader("🔍 Quick Model Evaluation")
        st.info("Evaluating all models to select the best performers...")
        
        ml_pipeline = ExpertMLPipeline()
        all_ml_models = ml_pipeline.get_all_models(task_type)
        
        evaluation_results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use a smaller subset for quick evaluation
        if len(X_train) > 1000:
            X_eval = X_train[:1000]
            y_eval = y_train[:1000]
            X_test_eval = X_test[:500]
            y_test_eval = y_test[:500]
        else:
            X_eval = X_train
            y_eval = y_train
            X_test_eval = X_test
            y_test_eval = y_test
        
        for idx, (model_name, model) in enumerate(all_ml_models.items()):
            status_text.text(f'⚡ Quick evaluating {model_name}...')
            progress_bar.progress((idx + 1) / len(all_ml_models))
            
            try:
                start_time = time.time()
                
                # Train model
                model.fit(X_eval, y_eval)
                
                # Make predictions
                if task_type == 'classification':
                    y_pred = model.predict(X_test_eval)
                    y_pred_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test_eval, y_pred)
                    precision = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                    roc_auc = roc_auc_score(y_test_eval, y_pred_proba) if y_pred_proba is not None else None
                    
                    evaluation_results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'evaluation_time': time.time() - start_time
                    }
                    
                else:
                    y_pred = model.predict(X_test_eval)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test_eval, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test_eval, y_pred)
                    r2 = r2_score(y_test_eval, y_pred)
                    
                    evaluation_results[model_name] = {
                        'model': model,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2_score': r2,
                        'evaluation_time': time.time() - start_time
                    }
                
            except Exception as e:
                st.warning(f"⚠️ Error evaluating {model_name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return evaluation_results
    
    def quick_evaluate_dl_models(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                               task_type='classification', epochs=20):
        """Quickly evaluate DL models to select the best ones"""
        st.subheader("🔍 Quick DL Model Evaluation")
        
        # Prepare data for DL
        if task_type == 'classification':
            if len(np.unique(y_train)) == 2:
                output_activation = 'sigmoid'
                y_train_dl = y_train
                y_val_dl = y_val
                y_test_dl = y_test
            else:
                output_activation = 'softmax'
                y_train_dl = to_categorical(y_train)
                y_val_dl = to_categorical(y_val)
                y_test_dl = to_categorical(y_test)
        else:
            output_activation = 'linear'
            y_train_dl = y_train
            y_val_dl = y_val
            y_test_dl = y_test
        
        input_shape = (X_train.shape[1],)
        all_dl_models = self.dl_models.get_all_dl_models(input_shape, task_type)
        
        evaluation_results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use smaller subset for quick evaluation
        if len(X_train) > 1000:
            X_train_eval = X_train[:1000]
            y_train_dl_eval = y_train_dl[:1000] if not isinstance(y_train_dl, (pd.Series, np.ndarray)) else y_train_dl[:1000]
            X_val_eval = X_val[:200]
            y_val_dl_eval = y_val_dl[:200] if not isinstance(y_val_dl, (pd.Series, np.ndarray)) else y_val_dl[:200]
            X_test_eval = X_test[:200]
            y_test_eval = y_test[:200]
        else:
            X_train_eval = X_train
            y_train_dl_eval = y_train_dl
            X_val_eval = X_val
            y_val_dl_eval = y_val_dl
            X_test_eval = X_test
            y_test_eval = y_test
        
        for idx, (model_name, model) in enumerate(all_dl_models.items()):
            status_text.text(f'⚡ Quick evaluating {model_name}...')
            progress_bar.progress((idx + 1) / len(all_dl_models))
            
            try:
                start_time = time.time()
                
                # Train model with fewer epochs for quick evaluation
                history = model.fit(
                    X_train_eval, y_train_dl_eval,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_val_eval, y_val_dl_eval),
                    verbose=0
                )
                
                # Make predictions
                if task_type == 'classification':
                    y_pred_proba = model.predict(X_test_eval)
                    
                    if output_activation == 'sigmoid':
                        y_pred = (y_pred_proba > 0.5).astype(int).ravel()
                        roc_auc = roc_auc_score(y_test_eval, y_pred_proba)
                    else:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        roc_auc = roc_auc_score(y_test_dl, y_pred_proba, multi_class='ovr')
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test_eval, y_pred)
                    precision = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)
                    
                    evaluation_results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'evaluation_time': time.time() - start_time,
                        'history': history.history
                    }
                    
                else:
                    y_pred = model.predict(X_test_eval).ravel()
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test_eval, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test_eval, y_pred)
                    r2 = r2_score(y_test_eval, y_pred)
                    
                    evaluation_results[model_name] = {
                        'model': model,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2_score': r2,
                        'evaluation_time': time.time() - start_time,
                        'history': history.history
                    }
                
            except Exception as e:
                st.warning(f"⚠️ Error evaluating {model_name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return evaluation_results
    
    def train_selected_ml_models(self, X_train, y_train, X_test, y_test, 
                               selected_models, task_type='classification'):
        """Train selected ML models with full training"""
        
        st.subheader("🤖 Training Selected ML Models")
        
        all_models = self.ml_pipeline.get_all_models(task_type)
        models_to_train = {name: all_models[name] for name in selected_models if name in all_models}
        
        if not models_to_train:
            st.error("❌ No valid models selected!")
            return {}
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (model_name, model) in enumerate(models_to_train.items()):
            status_text.text(f'🔄 Training {model_name}...')
            progress_bar.progress((idx + 1) / len(models_to_train))
            
            try:
                start_time = time.time()
                
                # Train model with full data
                model.fit(X_train, y_train)
                
                # Make predictions
                if task_type == 'classification':
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                    
                    results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'training_time': time.time() - start_time
                    }
                    
                else:
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    
                    results[model_name] = {
                        'model': model,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2_score': r2,
                        'mape': mape,
                        'training_time': time.time() - start_time
                    }
                
                st.success(f"✅ {model_name} trained successfully!")
                
            except Exception as e:
                st.warning(f"⚠️ Error training {model_name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def train_selected_dl_models(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                               selected_models, task_type='classification', epochs=100):
        """Train selected deep learning models with full training"""
        
        st.subheader("🧠 Training Selected Deep Learning Models")
        
        # Prepare data for DL
        if task_type == 'classification':
            if len(np.unique(y_train)) == 2:
                output_activation = 'sigmoid'
                loss = 'binary_crossentropy'
                y_train_dl = y_train
                y_val_dl = y_val
                y_test_dl = y_test
            else:
                output_activation = 'softmax'
                loss = 'categorical_crossentropy'
                y_train_dl = to_categorical(y_train)
                y_val_dl = to_categorical(y_val)
                y_test_dl = to_categorical(y_test)
        else:
            output_activation = 'linear'
            loss = 'mse'
            y_train_dl = y_train
            y_val_dl = y_val
            y_test_dl = y_test
        
        input_shape = (X_train.shape[1],)
        all_models = self.dl_models.get_all_dl_models(input_shape, task_type)
        models_to_train = {name: all_models[name] for name in selected_models if name in all_models}
        
        if not models_to_train:
            st.error("❌ No valid DL models selected!")
            return {}
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (model_name, model) in enumerate(models_to_train.items()):
            status_text.text(f'🧠 Training {model_name}...')
            progress_bar.progress((idx + 1) / len(models_to_train))
            
            try:
                start_time = time.time()
                
                # Train model with full data and epochs
                history = model.fit(
                    X_train, y_train_dl,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_val, y_val_dl),
                    callbacks=self.create_advanced_callbacks(),
                    verbose=0
                )
                
                # Store training history
                self.training_history[model_name] = history.history
                
                # Make predictions
                if task_type == 'classification':
                    y_pred_proba = model.predict(X_test)
                    
                    if output_activation == 'sigmoid':
                        y_pred = (y_pred_proba > 0.5).astype(int).ravel()
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                    else:
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        roc_auc = roc_auc_score(y_test_dl, y_pred_proba, multi_class='ovr')
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'training_time': time.time() - start_time,
                        'history': history.history
                    }
                    
                else:
                    y_pred = model.predict(X_test).ravel()
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    
                    results[model_name] = {
                        'model': model,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2_score': r2,
                        'mape': mape,
                        'training_time': time.time() - start_time,
                        'history': history.history
                    }
                
                st.success(f"✅ {model_name} trained successfully!")
                
            except Exception as e:
                st.warning(f"⚠️ Error training {model_name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results

class AdvancedDataVisualization:
    """Advanced Data Visualization with Interactive Controls"""
    
    def __init__(self):
        self.chart_data = {}
    
    def create_interactive_chart(self, df, chart_type, x_axis, y_axis, color_column=None, 
                               title=None, height=500):
        """Create interactive chart based on user selections"""
        
        try:
            if chart_type == "Scatter Plot":
                if color_column:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_column, 
                                   title=title, height=height)
                else:
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=title, height=height)
            
            elif chart_type == "Line Chart":
                if color_column:
                    fig = px.line(df, x=x_axis, y=y_axis, color=color_column, 
                                title=title, height=height)
                else:
                    fig = px.line(df, x=x_axis, y=y_axis, title=title, height=height)
            
            elif chart_type == "Bar Chart":
                if color_column:
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_column, 
                               title=title, height=height)
                else:
                    fig = px.bar(df, x=x_axis, y=y_axis, title=title, height=height)
            
            elif chart_type == "Histogram":
                if color_column:
                    fig = px.histogram(df, x=x_axis, color=color_column, 
                                     title=title, height=height, barmode='overlay')
                else:
                    fig = px.histogram(df, x=x_axis, title=title, height=height)
            
            elif chart_type == "Box Plot":
                if color_column:
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_column, 
                               title=title, height=height)
                else:
                    fig = px.box(df, x=x_axis, y=y_axis, title=title, height=height)
            
            elif chart_type == "Violin Plot":
                if color_column:
                    fig = px.violin(df, x=x_axis, y=y_axis, color=color_column, 
                                  title=title, height=height)
                else:
                    fig = px.violin(df, x=x_axis, y=y_axis, title=title, height=height)
            
            elif chart_type == "Heatmap":
                # For heatmap, we need to select only numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(corr_matrix, title="Correlation Heatmap", 
                                  height=height, aspect="auto")
                else:
                    st.warning("❌ Heatmap requires at least 2 numeric columns")
                    return None
            
            elif chart_type == "Pie Chart":
                if color_column:
                    fig = px.pie(df, names=x_axis, values=y_axis, color=color_column,
                               title=title, height=height)
                else:
                    fig = px.pie(df, names=x_axis, values=y_axis, title=title, height=height)
            
            elif chart_type == "Area Chart":
                if color_column:
                    fig = px.area(df, x=x_axis, y=y_axis, color=color_column,
                                title=title, height=height)
                else:
                    fig = px.area(df, x=x_axis, y=y_axis, title=title, height=height)
            
            else:
                st.error(f"❌ Chart type '{chart_type}' not supported")
                return None
            
            # Update layout for better appearance
            fig.update_layout(
                template='plotly_white',
                font=dict(size=12),
                title_font_size=20,
                showlegend=True
            )
            
            return fig
        
        except Exception as e:
            st.error(f"❌ Error creating chart: {str(e)}")
            return None
    
    def create_advanced_dashboard(self, df):
        """Create an advanced visualization dashboard"""
        
        st.markdown('<h2 class="sub-header">📊 Advanced Data Visualization Dashboard</h2>', unsafe_allow_html=True)
        
        # Store chart in session state to prevent reset
        if 'current_chart' not in st.session_state:
            st.session_state.current_chart = None
        
        # Chart configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_type = st.selectbox(
                "📈 Chart Type",
                ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
                 "Box Plot", "Violin Plot", "Heatmap", "Pie Chart", "Area Chart"]
            )
        
        with col2:
            available_columns = df.columns.tolist()
            x_axis = st.selectbox("X-Axis", available_columns, index=0)
            
            # For some charts, y-axis might not be needed
            if chart_type in ["Histogram", "Pie Chart"]:
                y_axis_options = [col for col in available_columns if col != x_axis]
                if y_axis_options:
                    y_axis = st.selectbox("Y-Axis/Values", y_axis_options, index=0)
                else:
                    y_axis = x_axis
            else:
                y_axis = st.selectbox("Y-Axis", [col for col in available_columns if col != x_axis], 
                                    index=min(1, len(available_columns)-1))
        
        with col3:
            color_options = ["None"] + available_columns
            color_column = st.selectbox("Color By", color_options, index=0)
            color_column = None if color_column == "None" else color_column
        
        # Additional options
        col4, col5 = st.columns(2)
        with col4:
            chart_title = st.text_input("Chart Title", f"{chart_type}: {x_axis} vs {y_axis}")
        with col5:
            chart_height = st.slider("Chart Height", 300, 800, 500)
        
        # Generate chart
        if st.button("🎨 Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                fig = self.create_interactive_chart(
                    df, chart_type, x_axis, y_axis, color_column, 
                    chart_title, chart_height
                )
                
                if fig:
                    st.session_state.current_chart = fig
                    st.plotly_chart(fig, use_container_width=True)
        
        # Download options for the current chart
        if st.session_state.current_chart is not None:
            st.subheader("💾 Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # HTML download
                html_string = st.session_state.current_chart.to_html()
                st.download_button(
                    label="📥 Download as HTML",
                    data=html_string,
                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
            
            with col2:
                # PNG download
                img_bytes = st.session_state.current_chart.to_image(format="png")
                st.download_button(
                    label="📷 Download as PNG",
                    data=img_bytes,
                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        
        # Quick visualization suggestions
        st.subheader("🚀 Quick Visualization Suggestions")
        
        col1, col2, col3 = st.columns(3)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        with col1:
            if st.button("📈 Correlation Heatmap") and len(numeric_cols) > 1:
                fig = self.create_interactive_chart(df, "Heatmap", "", "", None, "Correlation Heatmap", 500)
                if fig:
                    st.session_state.current_chart = fig
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("📊 Distribution Overview") and numeric_cols:
                fig = self.create_interactive_chart(df, "Histogram", numeric_cols[0], "", None, 
                                                  f"Distribution of {numeric_cols[0]}", 500)
                if fig:
                    st.session_state.current_chart = fig
                    st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if st.button("🎯 Category Analysis") and categorical_cols and numeric_cols:
                # Use the first categorical and first numeric column
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                fig = self.create_interactive_chart(df, "Box Plot", cat_col, num_col, 
                                                  None, f"{num_col} by {cat_col}", 500)
                if fig:
                    st.session_state.current_chart = fig
                    st.plotly_chart(fig, use_container_width=True)

class AdvancedAnalytics:
    """Advanced Analytics Module with Statistical Analysis"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def perform_comprehensive_analysis(self, df):
        """Perform comprehensive statistical analysis"""
        st.markdown('<h2 class="sub-header">🔍 Advanced Analytics & Statistical Analysis</h2>', unsafe_allow_html=True)
        
        # Let user choose which analyses to perform
        st.subheader("🎯 Select Analyses to Perform")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_stats = st.checkbox("Statistical Summary", value=True)
            show_correlation = st.checkbox("Correlation Analysis", value=True)
        with col2:
            show_distribution = st.checkbox("Distribution Analysis", value=True)
            show_outliers = st.checkbox("Outlier Detection", value=True)
        with col3:
            show_relationships = st.checkbox("Feature Relationships", value=True)
            show_clustering = st.checkbox("Cluster Analysis", value=False)
        
        # Basic dataset info
        if show_stats:
            st.subheader("📋 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            with col4:
                st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
            
            # Statistical Summary
            st.subheader("📊 Statistical Summary")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Correlation Analysis
        if show_correlation:
            st.subheader("🔗 Correlation Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              title="Correlation Heatmap",
                              color_continuous_scale='RdBu_r',
                              aspect="auto")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                st.write("**Top Positive Correlations:**")
                corr_pairs = corr_matrix.unstack()
                sorted_pairs = corr_pairs.sort_values(ascending=False)
                top_positive = sorted_pairs[sorted_pairs < 1].head(10)
                st.dataframe(top_positive, use_container_width=True)
        
        # Distribution Analysis
        if show_distribution:
            st.subheader("📈 Distribution Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Histogram
                    fig_hist = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Normality tests
                st.write("**Normality Tests:**")
                data = df[selected_col].dropna()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    try:
                        stat, p_value = normaltest(data)
                        st.metric("Normaltest p-value", f"{p_value:.4f}")
                    except:
                        st.write("Normaltest: N/A")
                
                with col2:
                    try:
                        stat, p_value = shapiro(data)
                        st.metric("Shapiro-Wilk p-value", f"{p_value:.4f}")
                    except:
                        st.write("Shapiro-Wilk: N/A")
                
                with col3:
                    try:
                        stat, p_value = jarque_bera(data)
                        st.metric("Jarque-Bera p-value", f"{p_value:.4f}")
                    except:
                        st.write("Jarque-Bera: N/A")
        
        # Outlier Detection
        if show_outliers:
            st.subheader("📊 Outlier Detection")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                outlier_col = st.selectbox("Select column for outlier detection:", numeric_cols, key="outlier_col")
                data = df[outlier_col].dropna()
                
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Outliers", len(outliers))
                with col2:
                    st.metric("Outlier Percentage", f"{(len(outliers)/len(data)*100):.2f}%")
                with col3:
                    st.metric("Outlier Range", f"[{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Feature Relationships
        if show_relationships:
            st.subheader("🔄 Feature Relationships")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("X Feature:", numeric_cols, index=0, key="x_feature")
                with col2:
                    y_feature = st.selectbox("Y Feature:", numeric_cols, index=min(1, len(numeric_cols)-1), key="y_feature")
                
                # Use simple scatter plot without trendline to avoid statsmodels dependency
                fig_scatter = px.scatter(df, x=x_feature, y=y_feature, 
                                       title=f"Relationship: {x_feature} vs {y_feature}")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Correlation stats
                correlation = df[x_feature].corr(df[y_feature])
                st.metric("Pearson Correlation", f"{correlation:.4f}")
        
        # Cluster Analysis
        if show_clustering:
            st.subheader("🔮 Cluster Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:
                # Use first two numeric columns for clustering
                X = df[numeric_cols[:2]].dropna()
                
                if len(X) > 0:
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    
                    # Create cluster visualization
                    X_clustered = X.copy()
                    X_clustered['Cluster'] = clusters
                    fig = px.scatter(
                        X_clustered, x=numeric_cols[0], y=numeric_cols[1], color='Cluster',
                        title="K-means Clustering (First 2 Features)",
                        hover_data=X_clustered.columns
                    )
                    st.plotly_chart(fig, use_container_width=True)

class ModelInterpretability:
    """Advanced Model Interpretability and Explainability"""
    
    def __init__(self):
        self.explanations = {}
        
    def calculate_feature_importance(self, model, X, y, feature_names, method='permutation'):
        """Calculate feature importance using multiple methods"""
        
        if method == 'permutation':
            # Permutation importance
            result = permutation_importance(
                model, X, y, 
                n_repeats=10,
                random_state=42,
                n_jobs=1
            )
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
        elif method == 'builtin' and hasattr(model, 'feature_importances_'):
            # Built-in feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        else:
            # Use SHAP-like approximation
            baseline_score = model.score(X, y)
            importance_scores = []
            
            for feature in range(X.shape[1]):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feature])
                permuted_score = model.score(X_permuted, y)
                importance_scores.append(baseline_score - permuted_score)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        return importance_df

class AdvancedModelEvaluation:
    """Advanced Model Evaluation and Comparison"""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def comprehensive_model_evaluation(self, models_results, X_test, y_test, task_type='classification'):
        """Perform comprehensive model evaluation"""
        
        st.subheader("📊 Comprehensive Model Evaluation")
        
        # Convert results to DataFrame
        if task_type == 'classification':
            metrics_df = pd.DataFrame({
                model_name: {
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'ROC-AUC': results.get('roc_auc', None),
                    'Training Time (s)': results['training_time']
                }
                for model_name, results in models_results.items()
            }).T
            
        else:
            metrics_df = pd.DataFrame({
                model_name: {
                    'MSE': results['mse'],
                    'RMSE': results['rmse'],
                    'MAE': results['mae'],
                    'R² Score': results['r2_score'],
                    'MAPE': results.get('mape', None),
                    'Training Time (s)': results['training_time']
                }
                for model_name, results in models_results.items()
            }).T
        
        # Display metrics
        st.dataframe(metrics_df.round(4), use_container_width=True)
        
        # Create comparison plots
        if task_type == 'classification':
            self.plot_classification_comparison(models_results)
        else:
            self.plot_regression_comparison(models_results, y_test)
        
        return metrics_df
    
    def plot_classification_comparison(self, models_results):
        """Plot classification model comparison"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy Comparison', 'Precision-Recall Comparison',
                          'F1-Score Comparison', 'Training Time Comparison'],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        model_names = list(models_results.keys())
        
        # Accuracy
        accuracies = [results['accuracy'] for results in models_results.values()]
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Accuracy'),
            row=1, col=1
        )
        
        # Precision-Recall
        precisions = [results['precision'] for results in models_results.values()]
        recalls = [results['recall'] for results in models_results.values()]
        fig.add_trace(
            go.Scatter(x=precisions, y=recalls, mode='markers+text',
                      text=model_names, textposition='top center',
                      marker=dict(size=12), name='Precision-Recall'),
            row=1, col=2
        )
        
        # F1-Score
        f1_scores = [results['f1_score'] for results in models_results.values()]
        fig.add_trace(
            go.Bar(x=model_names, y=f1_scores, name='F1-Score'),
            row=2, col=1
        )
        
        # Training Time
        training_times = [results['training_time'] for results in models_results.values()]
        fig.add_trace(
            go.Bar(x=model_names, y=training_times, name='Training Time (s)'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Classification Model Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_regression_comparison(self, models_results, y_test):
        """Plot regression model comparison"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['R² Score Comparison', 'Error Metrics Comparison',
                          'Prediction vs Actual', 'Training Time Comparison'],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        model_names = list(models_results.keys())
        
        # R² Score
        r2_scores = [results['r2_score'] for results in models_results.values()]
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R² Score'),
            row=1, col=1
        )
        
        # Error Metrics
        mses = [results['mse'] for results in models_results.values()]
        rmses = [results['rmse'] for results in models_results.values()]
        maes = [results['mae'] for results in models_results.values()]
        
        fig.add_trace(
            go.Bar(x=model_names, y=mses, name='MSE'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=model_names, y=rmses, name='RMSE'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=model_names, y=maes, name='MAE'),
            row=1, col=2
        )
        
        # Prediction vs Actual (for best model)
        best_model_name = max(models_results.keys(), 
                            key=lambda x: models_results[x]['r2_score'])
        best_model = models_results[best_model_name]['model']
        
        try:
            y_pred = best_model.predict(X_test)
            fig.add_trace(
                go.Scatter(x=y_test, y=y_pred, mode='markers',
                          name='Predictions', marker=dict(opacity=0.6)),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=[y_test.min(), y_test.max()], 
                          y=[y_test.min(), y_test.max()],
                          mode='lines', name='Ideal', line=dict(dash='dash')),
                row=2, col=1
            )
        except:
            pass
        
        # Training Time
        training_times = [results['training_time'] for results in models_results.values()]
        fig.add_trace(
            go.Bar(x=model_names, y=training_times, name='Training Time (s)'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Regression Model Comparison")
        st.plotly_chart(fig, use_container_width=True)

class ModelDeployment:
    """Advanced Model Deployment and Productionization"""
    
    def __init__(self):
        self.deployment_artifacts = {}
    
    def create_production_pipeline(self, best_model, feature_names, target_name, 
                                 task_type, scaler=None, encoder=None):
        """Create production-ready deployment pipeline"""
        
        # Fix the deployment code - remove the problematic format string
        feature_names_str = str(feature_names)
        
        deployment_code = f"""
# Production Model Deployment Pipeline
# Generated by Expert AutoML Platform
# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ProductionModel:
    \"\"\"Production-ready model deployment class\"\"\"
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_names = {feature_names_str}
        self.target_name = '{target_name}'
        self.task_type = '{task_type}'
        self.model_metadata = {{
            'version': '1.0.0',
            'created_date': '{datetime.now().strftime('%Y-%m-%d')}',
            'task_type': '{task_type}',
            'feature_count': {len(feature_names)}
        }}
    
    def load_artifacts(self, model_path, scaler_path=None, encoder_path=None):
        \"\"\"Load model and preprocessing artifacts\"\"\"
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            if scaler_path:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if encoder_path:
                with open(encoder_path, 'rb') as f:
                    self.encoder = pickle.load(f)
            
            print("✅ Model artifacts loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {{e}}")
            return False
    
    def validate_input(self, input_data):
        \"\"\"Validate input data\"\"\"
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input must be dictionary or DataFrame")
        
        # Check required features
        missing_features = set(self.feature_names) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {{missing_features}}")
        
        # Select and order features
        input_df = input_df[self.feature_names]
        
        return input_df
    
    def preprocess_input(self, input_data):
        \"\"\"Preprocess input data\"\"\"
        input_df = self.validate_input(input_data)
        
        # Handle missing values
        input_df = input_df.fillna(input_df.mean())
        
        # Scale features if scaler exists
        if self.scaler:
            input_df = pd.DataFrame(
                self.scaler.transform(input_df),
                columns=input_df.columns,
                index=input_df.index
            )
        
        return input_df
    
    def predict(self, input_data, return_probabilities=False, threshold=0.5):
        \"\"\"Make predictions\"\"\"
        if self.model is None:
            raise ValueError("Model not loaded. Call load_artifacts() first.")
        
        processed_data = self.preprocess_input(input_data)
        
        if self.task_type == 'classification':
            predictions = self.model.predict(processed_data)
            probabilities = None
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
            
            # Apply threshold if needed
            if return_probabilities and probabilities is not None:
                if probabilities.shape[1] == 2:  # Binary classification
                    positive_probs = probabilities[:, 1]
                    predictions = (positive_probs > threshold).astype(int)
                return {{
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'confidence': np.max(probabilities, axis=1).tolist()
                }}
            else:
                return {{
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist() if probabilities is not None else None
                }}
        
        else:  # regression
            predictions = self.model.predict(processed_data)
            return {{
                'predictions': predictions.tolist(),
                'confidence': None  # Not applicable for regression
            }}
    
    def batch_predict(self, input_file_path, output_file_path, **kwargs):
        \"\"\"Perform batch predictions on a CSV file\"\"\"
        try:
            # Read input data
            input_data = pd.read_csv(input_file_path)
            
            # Make predictions
            results = self.predict(input_data, **kwargs)
            
            # Add predictions to original data
            output_data = input_data.copy()
            output_data['predictions'] = results['predictions']
            
            if results.get('probabilities'):
                if isinstance(results['probabilities'][0], list):
                    # Multi-class probabilities
                    for i in range(len(results['probabilities'][0])):
                        output_data[f'probability_class_{{i}}'] = [p[i] for p in results['probabilities']]
                else:
                    # Binary classification probability
                    output_data['probability'] = results['probabilities']
            
            if results.get('confidence'):
                output_data['prediction_confidence'] = results['confidence']
            
            # Save results
            output_data.to_csv(output_file_path, index=False)
            print(f"✅ Batch predictions saved to {{output_file_path}}")
            
            return output_data
            
        except Exception as e:
            print(f"❌ Error in batch prediction: {{e}}")
            return None
    
    def get_model_info(self):
        \"\"\"Get model information and metadata\"\"\"
        info = self.model_metadata.copy()
        info.update({{
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'model_type': type(self.model).__name__,
            'has_scaler': self.scaler is not None,
            'has_encoder': self.encoder is not None
        }})
        return info

# Example usage
if __name__ == "__main__":
    # Initialize deployment
    deployment = ProductionModel()
    
    # Load model artifacts (update paths as needed)
    # deployment.load_artifacts(
    #     model_path='best_model.pkl',
    #     scaler_path='scaler.pkl',
    #     encoder_path='encoder.pkl'
    # )
    
    # Example prediction
    sample_input = {{
        {', '.join([f"'{feature}': 0.0" for feature in feature_names[:min(3, len(feature_names))]])}
        # ... add all feature values
    }}
    
    # result = deployment.predict(sample_input)
    # print("Prediction result:", result)
    
    # Batch prediction example
    # deployment.batch_predict('input_data.csv', 'predictions.csv')
    
    # Get model info
    # print("Model info:", deployment.get_model_info())
"""
        
        return deployment_code
    
    def save_deployment_artifacts(self, best_model, feature_names, scaler=None, 
                                encoder=None, model_name="best_model"):
        """Save all deployment artifacts"""
        
        artifacts = {}
        
        # Save model
        model_path = f"{model_name}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            artifacts['model'] = model_path
        except Exception as e:
            st.error(f"❌ Error saving model: {e}")
            return None
        
        # Save scaler if exists
        if scaler is not None:
            scaler_path = "scaler.pkl"
            try:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                artifacts['scaler'] = scaler_path
            except Exception as e:
                st.warning(f"⚠️ Error saving scaler: {e}")
        
        # Save encoder if exists
        if encoder is not None:
            encoder_path = "encoder.pkl"
            try:
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)
                artifacts['encoder'] = encoder_path
            except Exception as e:
                st.warning(f"⚠️ Error saving encoder: {e}")
        
        # Save feature names
        features_path = "feature_names.json"
        try:
            with open(features_path, 'w') as f:
                json.dump(feature_names, f)
            artifacts['feature_names'] = features_path
        except Exception as e:
            st.warning(f"⚠️ Error saving feature names: {e}")
        
        # Save metadata
        metadata = {
            'created_date': datetime.now().isoformat(),
            'model_type': type(best_model).__name__,
            'feature_count': len(feature_names),
            'artifacts_version': '1.0'
        }
        
        metadata_path = "model_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            artifacts['metadata'] = metadata_path
        except Exception as e:
            st.warning(f"⚠️ Error saving metadata: {e}")
        
        self.deployment_artifacts = artifacts
        return artifacts
    
    def get_deployment_guide(self, task_type, model_type):
        """Get deployment guide and suggestions"""
        
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Deployment Guide & Next Steps</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📋 Deployment Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🐍 Streamlit Deployment
            
            **Quick Setup:**
            ```python
            import streamlit as st
            import pickle
            import pandas as pd
            
            # Load your model
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            st.title('ML Model Deployment')
            
            # Create input fields
            features = []
            for feature in feature_names:
                val = st.number_input(feature, value=0.0)
                features.append(val)
            
            if st.button('Predict'):
                prediction = model.predict([features])
                st.success(f'Prediction: {prediction[0]}')
            ```
            
            **Deployment Platforms:**
            - Streamlit Sharing (Free)
            - Heroku
            - AWS EC2
            - Google Cloud Run
            """)
        
        with col2:
            st.markdown("""
            ### 🌐 REST API Deployment
            
            **FastAPI Example:**
            ```python
            from fastapi import FastAPI
            import pickle
            import pandas as pd
            
            app = FastAPI()
            
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            @app.post("/predict")
            async def predict(features: dict):
                input_data = [features[f] for f in feature_names]
                prediction = model.predict([input_data])
                return {"prediction": prediction[0]}
            ```
            
            **Deployment Options:**
            - Docker + Any cloud provider
            - AWS Lambda
            - Google Cloud Functions
            - Azure Functions
            """)
        
        st.subheader("🔧 Production Best Practices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Model Monitoring
            - Track prediction drift
            - Monitor feature distributions
            - Set up alerting for anomalies
            - Regular model retraining
            
            ### 🛡️ Security
            - Input validation
            - Rate limiting
            - API authentication
            - Data encryption
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Performance
            - Model caching
            - Batch predictions
            - Async processing for large datasets
            - Load balancing
            
            ### 🔄 CI/CD
            - Automated testing
            - Version control for models
            - Rollback strategies
            - A/B testing
            """)
        
        st.subheader("🎯 Next Steps")
        
        st.markdown(f"""
        1. **Test Locally** - Run the provided deployment code locally first
        2. **Containerize** - Create a Docker image for your model
        3. **Choose Platform** - Select deployment platform based on your needs
        4. **Monitor** - Set up monitoring and alerting
        5. **Scale** - Plan for scalability as usage grows
        
        **Recommended for {task_type} task:**
        - For prototypes: Streamlit Sharing
        - For production: FastAPI + Docker
        - For serverless: AWS Lambda/Google Cloud Functions
        """)

def detect_encoding(file):
    """Detect file encoding for proper CSV reading"""
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding'] or 'utf-8'

def advanced_data_profiling(df):
    """Generate comprehensive data profiling report"""
    st.markdown('<h2 class="sub-header">📊 Advanced Data Profiling</h2>', unsafe_allow_html=True)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Rows</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Columns</h3>
            <h2>{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Missing Values</h3>
            <h2>{df.isnull().sum().sum():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Duplicate Rows</h3>
            <h2>{df.duplicated().sum():,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Column statistics
    st.subheader("📋 Column Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    col_stats = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique(),
        'Unique Percentage': (df.nunique() / len(df) * 100).round(2)
    })
    
    st.dataframe(col_stats, use_container_width=True)
    
    return numeric_cols, categorical_cols, datetime_cols

def intelligent_data_cleaning(df, missing_thresh=0.5, fill_num_option='mean', 
                            fill_cat_option='mode', drop_outliers=False, encoding_method='auto'):
    """Advanced data cleaning with intelligent preprocessing"""
    st.markdown('<h2 class="sub-header">🧹 Intelligent Data Cleaning</h2>', unsafe_allow_html=True)
    
    df_cleaned = df.copy()
    cleaning_log = []
    
    # Remove duplicates
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    removed_duplicates = initial_rows - len(df_cleaned)
    if removed_duplicates > 0:
        cleaning_log.append(f"✅ Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values by column threshold
    missing_percent = df_cleaned.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > missing_thresh].index
    if len(cols_to_drop) > 0:
        df_cleaned = df_cleaned.drop(columns=cols_to_drop)
        cleaning_log.append(f"🗑️ Dropped {len(cols_to_drop)} columns with >{missing_thresh*100}% missing values: {list(cols_to_drop)}")
    
    # Identify column types
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    # Fill missing values in numeric columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().sum() > 0:
            if fill_num_option == 'mean':
                fill_value = df_cleaned[col].mean()
            elif fill_num_option == 'median':
                fill_value = df_cleaned[col].median()
            elif fill_num_option == 'mode':
                fill_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else df_cleaned[col].mean()
            else:
                fill_value = 0
            
            df_cleaned[col].fillna(fill_value, inplace=True)
            cleaning_log.append(f"🔧 Filled missing values in '{col}' with {fill_num_option}: {fill_value:.2f}")
    
    # Fill missing values in categorical columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().sum() > 0:
            if fill_cat_option == 'mode':
                fill_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
            else:
                fill_value = fill_cat_option
            
            df_cleaned[col].fillna(fill_value, inplace=True)
            cleaning_log.append(f"🔧 Filled missing values in '{col}' with: {fill_value}")
    
    # Outlier detection and removal
    if drop_outliers and numeric_cols:
        initial_rows = len(df_cleaned)
        for col in numeric_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            outliers_removed = outlier_mask.sum()
            df_cleaned = df_cleaned[~outlier_mask]
            
            if outliers_removed > 0:
                cleaning_log.append(f"📊 Removed {outliers_removed} outliers from '{col}'")
        
        total_outliers_removed = initial_rows - len(df_cleaned)
        if total_outliers_removed > 0:
            cleaning_log.append(f"🎯 Total outliers removed: {total_outliers_removed}")
    
    # Intelligent encoding
    if encoding_method == 'auto':
        # Decide encoding method based on cardinality
        for col in categorical_cols:
            unique_values = df_cleaned[col].nunique()
            if unique_values > 10:  # Use label encoding for high cardinality
                le = LabelEncoder()
                df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
                cleaning_log.append(f"🏷️ Applied Label Encoding to '{col}' ({unique_values} unique values)")
            else:  # Use one-hot encoding for low cardinality
                dummies = pd.get_dummies(df_cleaned[col], prefix=col, drop_first=True)
                df_cleaned = df_cleaned.drop(col, axis=1)
                df_cleaned = pd.concat([df_cleaned, dummies], axis=1)
                cleaning_log.append(f"🎯 Applied One-Hot Encoding to '{col}' ({unique_values} unique values)")
    
    # Display cleaning log
    st.subheader("🔍 Cleaning Summary")
    for log_entry in cleaning_log:
        st.write(log_entry)
    
    # Show before/after comparison
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Shape", f"{df.shape[0]} × {df.shape[1]}")
    with col2:
        st.metric("Cleaned Shape", f"{df_cleaned.shape[0]} × {df_cleaned.shape[1]}")
    
    # ADDED: Download cleaned data option
    st.subheader("💾 Download Cleaned Data")
    csv = df_cleaned.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned Data as CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
    
    return df_cleaned

def main():
    """Main application function"""
    
    st.markdown('<h1 class="main-header">🧠 Expert AutoML & Deep Learning Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Enterprise-grade Machine Learning & Deep Learning Platform with Advanced Model Training, 
            Interpretability, and Production Deployment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ExpertModelTrainer()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = AdvancedModelEvaluation()
    if 'deployment' not in st.session_state:
        st.session_state.deployment = ModelDeployment()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = AdvancedDataVisualization()
    if 'analytics' not in st.session_state:
        st.session_state.analytics = AdvancedAnalytics()
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    if 'dl_evaluation_results' not in st.session_state:
        st.session_state.dl_evaluation_results = {}
    
    # Sidebar configuration
    st.sidebar.header('⚙️ Expert Configuration')
    
    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset (CSV, XLSX, XLS)", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to begin expert-level analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            with st.spinner('📊 Loading data...'):
                if uploaded_file.name.endswith('.csv'):
                    encoding = detect_encoding(uploaded_file)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.session_state.original_data = df.copy()
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Show raw data sample
            with st.expander("👀 View Raw Data Sample", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)
            
            # Data profiling
            numeric_cols, categorical_cols, datetime_cols = advanced_data_profiling(df)
            
            # Sidebar cleaning options
            st.sidebar.subheader('🧹 Data Cleaning Options')
            missing_threshold = st.sidebar.slider("Missing value threshold (%)", 0, 100, 50) / 100
            numerical_fill = st.sidebar.selectbox("Fill numeric missing values with:", 
                                                 ['mean', 'median', 'mode', 'zero'])
            categorical_fill = st.sidebar.selectbox("Fill categorical missing values with:", 
                                                   ['mode', 'unknown', 'most_frequent'])
            remove_outliers = st.sidebar.checkbox("Remove outliers (IQR method)", value=False)
            encoding_method = st.sidebar.selectbox("Encoding method:", ['auto', 'label', 'onehot'])
            
            # Data cleaning
            if st.button("🧹 Clean Data", type="primary"):
                with st.spinner('🔄 Cleaning data...'):
                    st.session_state.processed_data = intelligent_data_cleaning(
                        df, missing_threshold, numerical_fill, categorical_fill, 
                        remove_outliers, encoding_method
                    )
                
                st.success("✅ Data cleaning completed!")
                
                # Show cleaned data sample
                with st.expander("👀 View Cleaned Data Sample", expanded=True):
                    st.dataframe(st.session_state.processed_data.head(100), use_container_width=True)
            
            # Application mode selection
            current_data = st.session_state.processed_data if st.session_state.processed_data is not None else df
            
            st.sidebar.subheader('🎯 Expert Analysis Mode')
            app_mode = st.sidebar.radio(
                "Choose your analysis type:",
                ["📊 Data Visualization", "🤖 Expert Model Training", "🔍 Advanced Analytics", "🚀 Model Deployment"]
            )
            
            if app_mode == "📊 Data Visualization":
                st.session_state.visualizer.create_advanced_dashboard(current_data)
            
            elif app_mode == "🔍 Advanced Analytics":
                st.session_state.analytics.perform_comprehensive_analysis(current_data)
            
            elif app_mode == "🤖 Expert Model Training":
                st.markdown('<h2 class="sub-header">🎯 Expert Machine Learning & Deep Learning</h2>', unsafe_allow_html=True)
                
                # Feature and target selection
                all_columns = current_data.columns.tolist()
                numeric_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    target_column = st.selectbox("🎯 Select target variable:", all_columns)
                
                with col2:
                    # Auto-detect task type
                    if target_column in numeric_columns:
                        unique_values = current_data[target_column].nunique()
                        if unique_values <= 10 and unique_values >= 2:
                            default_task = "classification"
                        else:
                            default_task = "regression"
                    else:
                        default_task = "classification"
                    
                    task_type = st.selectbox("📋 Task type:", 
                                           ["classification", "regression"],
                                           index=0 if default_task == "classification" else 1)
                
                # Feature selection
                available_features = [col for col in all_columns if col != target_column]
                selected_features = st.multiselect(
                    "🔧 Select feature columns:",
                    available_features,
                    default=available_features[:min(20, len(available_features))]
                )
                
                # Model selection mode
                st.subheader("🤖 Model Selection Strategy")
                model_selection_mode = st.radio(
                    "Choose model selection approach:",
                    ["🚀 Auto Recommendation (Evaluate & Select Best Models)", "👨‍💻 Manual Selection (Choose Specific Models)"],
                    horizontal=True
                )
                
                # Advanced options
                with st.expander("⚙️ Expert Training Options", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        test_size = st.slider("Test set size", 0.1, 0.4, 0.2)
                        validation_size = st.slider("Validation size", 0.1, 0.3, 0.1)
                        random_state = st.number_input("Random state", value=42)
                    
                    with col2:
                        scale_features = st.checkbox("Scale features", value=True)
                        feature_selection = st.checkbox("Apply feature selection", value=False)
                        advanced_preprocessing = st.checkbox("Advanced preprocessing", value=True)
                    
                    with col3:
                        train_ml = st.checkbox("Train ML Models", value=True)
                        train_dl = st.checkbox("Train DL Models", value=False)
                        epochs = st.number_input("DL Epochs", min_value=10, max_value=500, value=100) if train_dl else 100
                
                # Model training and evaluation
                if st.button("🚀 Start Expert Model Training", type="primary"):
                    if len(selected_features) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        with st.spinner('🔄 Training expert models...'):
                            # Prepare data
                            X = current_data[selected_features]
                            y = current_data[target_column]
                            
                            # Handle missing values in target
                            if y.isnull().sum() > 0:
                                st.warning(f"⚠️ Removing {y.isnull().sum()} rows with missing target values.")
                                valid_indices = y.notna()
                                X = X[valid_indices]
                                y = y[valid_indices]
                            
                            # Feature scaling
                            scaler = None
                            if scale_features:
                                scaler = StandardScaler()
                                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                            
                            # Feature selection
                            if feature_selection:
                                if task_type == 'classification':
                                    selector = SelectKBest(f_classif, k=min(15, X.shape[1]))
                                else:
                                    selector = SelectKBest(f_regression, k=min(15, X.shape[1]))
                                
                                X = pd.DataFrame(selector.fit_transform(X, y), 
                                               columns=X.columns[selector.get_support()], 
                                               index=X.index)
                                
                                st.info(f"🔍 Selected {X.shape[1]} most important features")
                            
                            # Train-validation-test split
                            X_temp, X_test, y_temp, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state,
                                stratify=y if task_type == 'classification' and len(y.unique()) > 1 else None
                            )
                            
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_temp, y_temp, test_size=validation_size, random_state=random_state,
                                stratify=y_temp if task_type == 'classification' and len(y_temp.unique()) > 1 else None
                            )
                            
                            st.info(f"📊 Training set: {X_train.shape[0]} samples, "
                                  f"Validation set: {X_val.shape[0]} samples, "
                                  f"Test set: {X_test.shape[0]} samples")
                            
                            # Train models based on selection mode
                            all_results = {}
                            
                            if model_selection_mode == "🚀 Auto Recommendation (Evaluate & Select Best Models)":
                                st.subheader("🔍 Auto Model Evaluation & Selection")
                                
                                # Quick evaluate ML models
                                if train_ml:
                                    st.info("⚡ Quick evaluating ML models to select the best performers...")
                                    ml_evaluation_results = st.session_state.trainer.quick_evaluate_models(
                                        X_train, y_train, X_test, y_test, task_type
                                    )
                                    st.session_state.evaluation_results = ml_evaluation_results
                                    
                                    if ml_evaluation_results:
                                        # Select best ML models
                                        ml_pipeline = ExpertMLPipeline()
                                        best_ml_models = ml_pipeline.get_best_models_auto(ml_evaluation_results, task_type)
                                        
                                        st.success(f"🎯 Auto-selected ML Models: {', '.join(best_ml_models.keys())}")
                                        
                                        # Train the best ML models with full data
                                        ml_results = st.session_state.trainer.train_selected_ml_models(
                                            X_train, y_train, X_test, y_test, 
                                            list(best_ml_models.keys()), task_type
                                        )
                                        all_results.update(ml_results)
                                
                                # Quick evaluate DL models
                                if train_dl:
                                    st.info("⚡ Quick evaluating DL models to select the best performers...")
                                    dl_evaluation_results = st.session_state.trainer.quick_evaluate_dl_models(
                                        X_train, y_train, X_val, y_val, X_test, y_test, 
                                        task_type, epochs=20
                                    )
                                    st.session_state.dl_evaluation_results = dl_evaluation_results
                                    
                                    if dl_evaluation_results:
                                        # Select best DL models
                                        dl_models = AdvancedDeepLearningModels()
                                        best_dl_models = dl_models.get_best_dl_models_auto(dl_evaluation_results, task_type)
                                        
                                        st.success(f"🎯 Auto-selected DL Models: {', '.join(best_dl_models.keys())}")
                                        
                                        # Train the best DL models with full data
                                        dl_results = st.session_state.trainer.train_selected_dl_models(
                                            X_train, y_train, X_val, y_val, X_test, y_test, 
                                            list(best_dl_models.keys()), task_type, epochs
                                        )
                                        all_results.update(dl_results)
                            
                            else:  # Manual selection
                                st.subheader("👨‍💻 Manual Model Selection")
                                
                                ml_pipeline = ExpertMLPipeline()
                                dl_models = AdvancedDeepLearningModels()
                                
                                if train_ml:
                                    # Get all available ML models
                                    all_ml_models = ml_pipeline.get_all_models(task_type)
                                    ml_models_to_train = st.multiselect(
                                        "Select ML Models to train:",
                                        list(all_ml_models.keys()),
                                        default=["Random Forest", "XGBoost", "LightGBM"]
                                    )
                                    
                                    if ml_models_to_train:
                                        ml_results = st.session_state.trainer.train_selected_ml_models(
                                            X_train, y_train, X_test, y_test, 
                                            ml_models_to_train, task_type
                                        )
                                        all_results.update(ml_results)
                                
                                if train_dl:
                                    input_shape = (X_train.shape[1],)
                                    all_dl_models = dl_models.get_all_dl_models(input_shape, task_type)
                                    dl_models_to_train = st.multiselect(
                                        "Select Deep Learning Models to train:",
                                        list(all_dl_models.keys()),
                                        default=["Simple Neural Network", "Residual MLP"]
                                    )
                                    
                                    if dl_models_to_train:
                                        dl_results = st.session_state.trainer.train_selected_dl_models(
                                            X_train, y_train, X_val, y_val, X_test, y_test, 
                                            dl_models_to_train, task_type, epochs
                                        )
                                        all_results.update(dl_results)
                            
                            if all_results:
                                st.session_state.model_results = all_results
                                
                                # Comprehensive evaluation
                                metrics_df = st.session_state.evaluator.comprehensive_model_evaluation(
                                    all_results, X_test, y_test, task_type
                                )
                                
                                # Find best model
                                if task_type == 'classification':
                                    best_model_name = max(all_results.keys(), 
                                                       key=lambda x: all_results[x]['accuracy'])
                                else:
                                    best_model_name = max(all_results.keys(), 
                                                       key=lambda x: all_results[x]['r2_score'])
                                
                                best_model = all_results[best_model_name]['model']
                                st.session_state.best_model = best_model
                                st.session_state.best_model_name = best_model_name
                                st.session_state.feature_names = selected_features
                                st.session_state.target_name = target_column
                                st.session_state.task_type = task_type
                                st.session_state.scaler = scaler
                                
                                st.markdown(f"""
                                <div class="success-box">
                                    <h3>🏆 Best Model: {best_model_name}</h3>
                                    <p><strong>Model Type:</strong> {type(best_model).__name__}</p>
                                    {f"<p><strong>Accuracy:</strong> {all_results[best_model_name]['accuracy']:.4f}</p>" if task_type == 'classification' else f"<p><strong>R² Score:</strong> {all_results[best_model_name]['r2_score']:.4f}</p>"}
                                    <p><strong>Training Time:</strong> {all_results[best_model_name]['training_time']:.2f} seconds</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            else:
                                st.error("❌ No models were successfully trained.")
            
            elif app_mode == "🚀 Model Deployment":
                st.markdown('<h2 class="sub-header">🚀 Production Model Deployment</h2>', unsafe_allow_html=True)
                
                if st.session_state.best_model is not None:
                    # Deployment options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📦 Deployment Artifacts")
                        
                        if st.button("💾 Save Deployment Artifacts"):
                            artifacts = st.session_state.deployment.save_deployment_artifacts(
                                st.session_state.best_model,
                                st.session_state.feature_names,
                                st.session_state.scaler,
                                None,  # encoder
                                st.session_state.best_model_name
                            )
                            
                            if artifacts:
                                st.success("✅ Deployment artifacts saved successfully!")
                                
                                # ADDED: Direct model download
                                st.subheader("📥 Download Model")
                                with open(artifacts['model'], 'rb') as f:
                                    model_bytes = f.read()
                                
                                st.download_button(
                                    label="💾 Download Model as PKL",
                                    data=model_bytes,
                                    file_name=f"{st.session_state.best_model_name}.pkl",
                                    mime="application/octet-stream"
                                )
                    
                    with col2:
                        st.subheader("🔧 Deployment Code")
                        
                        # Generate deployment code
                        deployment_code = st.session_state.deployment.create_production_pipeline(
                            st.session_state.best_model,
                            st.session_state.feature_names,
                            st.session_state.target_name,
                            st.session_state.task_type,
                            st.session_state.scaler
                        )
                        
                        st.code(deployment_code, language='python')
                        
                        # Download deployment code
                        st.download_button(
                            label="📥 Download Deployment Code",
                            data=deployment_code,
                            file_name=f"{st.session_state.best_model_name}_deployment.py",
                            mime="text/plain"
                        )
                    
                    # Model interpretability
                    st.subheader("🔍 Model Interpretability")
                    
                    if st.button("📊 Generate Model Explanations"):
                        with st.spinner('Generating model explanations...'):
                            # Prepare interpretability data
                            X_interpret = current_data[st.session_state.feature_names]
                            y_interpret = current_data[st.session_state.target_name]
                            
                            interpretability = ModelInterpretability()
                            
                            importance_df = interpretability.calculate_feature_importance(
                                st.session_state.best_model,
                                X_interpret.values,
                                y_interpret.values,
                                np.array(st.session_state.feature_names)
                            )
                            
                            # Plot feature importance
                            fig = px.bar(
                                importance_df.head(15),
                                x='importance_mean' if 'importance_mean' in importance_df.columns else 'importance',
                                y='feature',
                                orientation='h',
                                title=f"Feature Importance - {st.session_state.best_model_name}",
                                labels={'importance_mean': 'Importance', 'feature': 'Feature'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show feature importance table
                            st.dataframe(importance_df, use_container_width=True)
                    
                    # ADDED: Deployment Guide
                    st.session_state.deployment.get_deployment_guide(
                        st.session_state.task_type,
                        type(st.session_state.best_model).__name__
                    )
                
                else:
                    st.warning("⚠️ Please train models first in the 'Expert Model Training' section.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
    
    else:
        # Landing page content
        st.markdown("""
        ## 🌟 Expert-Level Features
        
        ### 🤖 **Advanced Machine Learning**
        - **Auto Model Evaluation & Selection** - Evaluates all models and selects best performers
        - **Manual Model Selection** - Full control over model choices
        - **Quick Model Evaluation** - Fast performance assessment before full training
        - **Optimized Hyperparameter Tuning**
        
        ### 🧠 **Deep Learning Architectures**
        - **Transformer Models** for tabular data
        - **Residual MLP Networks** with skip connections
        - **Hybrid CNN-LSTM Models** for sequential patterns
        - **Simple Neural Networks** for quick prototyping
        
        ### 📊 **Advanced Data Visualization**
        - **Interactive Chart Builder** with full customization
        - **Multiple Chart Types** (Scatter, Line, Bar, Histogram, Box, Violin, Heatmap, Pie, Area)
        - **Color Coding** and advanced styling options
        - **Quick Visualization Suggestions** for instant insights
        
        ### 🔍 **Advanced Analytics**
        - **Customizable Analysis** - Choose which analyses to perform
        - **Statistical Analysis** with comprehensive insights
        - **Correlation Analysis** with heatmaps
        - **Distribution Analysis** with normality tests
        - **Outlier Detection** and analysis
        
        ---
        
        ## 🎯 **Auto Model Selection Process**
        
        **1. Quick Evaluation Phase ⚡**
        - All models are quickly evaluated on a subset of data
        - Performance metrics are calculated for each model
        - Best performing models are automatically selected
        
        **2. Full Training Phase 🚀**
        - Only the best models are trained with full data
        - Saves time and computational resources
        - Ensures optimal model performance
        
        **3. Comprehensive Evaluation 📊**
        - Detailed performance comparison
        - Model interpretability and feature importance
        - Production-ready deployment code
        
        ---
        
        ## 🚀 **Get Started**
        
        Upload your dataset to access these expert-level features:
        
        1. **📊 Data Visualization** - Interactive charts and insights
        2. **🔍 Advanced Analytics** - Customizable statistical analysis
        3. **🤖 Model Training** - Auto evaluation & selection or manual model training
        4. **🚀 Deployment** - Production-ready model deployment with comprehensive guides
        
        **Ready to build enterprise-grade AI solutions? Upload your data now! 📂✨**
        """)

if __name__ == "__main__":
    main()