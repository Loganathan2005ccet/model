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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, LSTM, GRU, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
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
                           mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from scipy.stats import normaltest, shapiro, jarque_bera, anderson
import time
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# 🔧 FIX: Environment settings to prevent multiprocessing issues
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Set page configuration
st.set_page_config(
    page_title="Advanced AutoML & Visualization Dashboard",
    page_icon="🚀",
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
</style>
""", unsafe_allow_html=True)

def detect_encoding(file):
    """Detect file encoding for proper CSV reading"""
    raw_data = file.read(10000)
    result = chardet.detect(raw_data)
    file.seek(0)
    return result['encoding'] or 'utf-8'

def advanced_data_profiling(df):
    """Generate comprehensive data profiling report"""
    st.markdown('<h2 class="sub-header">📊 Advanced Data Profiling</h2>', unsafe_allow_html=True)
    
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
    
    # Distribution analysis for numeric columns
    if numeric_cols:
        st.subheader("📈 Numeric Columns Distribution")
        for col in numeric_cols[:5]:  # Show first 5 numeric columns
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=col, title=f'Distribution of {col}', 
                                 color_discrete_sequence=['#667eea'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=col, title=f'Box Plot of {col}',
                           color_discrete_sequence=['#764ba2'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Download cleaned dataset
    st.subheader("📥 Download Cleaned Dataset")
    csv_data = df_cleaned.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )
    
    return df_cleaned

def create_advanced_visualizations(df, sample_size=5000):
    """Create comprehensive interactive visualizations with individual chart selection"""
    st.markdown('<h2 class="sub-header">📊 Advanced Interactive Visualizations</h2>', unsafe_allow_html=True)
    
    # Sample data for performance
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        st.info(f"📊 Visualizing a sample of {sample_size:,} rows out of {len(df):,} total rows for performance.")
    else:
        df_sample = df.copy()
    
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df_sample.select_dtypes(include=['datetime64']).columns.tolist()
    all_cols = df_sample.columns.tolist()
    
    # Sidebar for visualization controls
    st.sidebar.header('🎨 Visualization Controls')
    
    # Chart type selection
    chart_types = {
        "Scatter Plot": "scatter",
        "Line Chart": "line", 
        "Bar Chart": "bar",
        "Histogram": "histogram",
        "Box Plot": "box",
        "Violin Plot": "violin",
        "Heatmap": "heatmap",
        "Area Chart": "area",
        "Pie Chart": "pie",
        "Sunburst Chart": "sunburst",
        "Treemap": "treemap",
        "3D Scatter Plot": "scatter_3d",
        "Correlation Matrix": "correlation",
        "Distribution Plot": "distribution",
        "Pair Plot": "pair_plot",
        "Parallel Coordinates": "parallel",
        "Radar Chart": "radar",
        "Bubble Chart": "bubble",
        "Density Heatmap": "density_heatmap",
        "Statistical Summary": "stats_summary"
    }
    
    selected_chart = st.sidebar.selectbox(
        "Select Chart Type:",
        list(chart_types.keys())
    )
    
    chart_type = chart_types[selected_chart]
    
    # Dynamic axis selection based on chart type
    st.sidebar.subheader('📊 Chart Configuration')
    
    # Color scheme selection
    color_schemes = {
        'Viridis': px.colors.sequential.Viridis,
        'Plasma': px.colors.sequential.Plasma,
        'Blues': px.colors.sequential.Blues,
        'Reds': px.colors.sequential.Reds,
        'Plotly': px.colors.qualitative.Plotly,
        'Set1': px.colors.qualitative.Set1,
        'Custom': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    }
    
    selected_color_scheme = st.sidebar.selectbox("Color Scheme:", list(color_schemes.keys()))
    colors = color_schemes[selected_color_scheme]
    
    # Chart-specific configurations
    if chart_type in ["scatter", "line", "area", "bubble"]:
        x_axis = st.sidebar.selectbox("X-axis:", all_cols, key='x_axis')
        y_axis = st.sidebar.selectbox("Y-axis:", numeric_cols, key='y_axis')
        color_col = st.sidebar.selectbox("Color by (optional):", [None] + all_cols, key='color')
        
        if chart_type == "bubble":
            size_col = st.sidebar.selectbox("Size by:", numeric_cols, key='size')
        
    elif chart_type == "bar":
        x_axis = st.sidebar.selectbox("X-axis (Categories):", all_cols, key='bar_x')
        y_axis = st.sidebar.selectbox("Y-axis (Values):", [None] + numeric_cols, key='bar_y')
        color_col = st.sidebar.selectbox("Color by (optional):", [None] + all_cols, key='bar_color')
        
    elif chart_type == "histogram":
        x_axis = st.sidebar.selectbox("Column for Histogram:", numeric_cols, key='hist_x')
        bins = st.sidebar.slider("Number of Bins:", 10, 100, 30)
        color_col = st.sidebar.selectbox("Color by (optional):", [None] + all_cols, key='hist_color')
        
    elif chart_type in ["box", "violin"]:
        x_axis = st.sidebar.selectbox("X-axis (Categories):", [None] + all_cols, key='box_x')
        y_axis = st.sidebar.selectbox("Y-axis (Values):", numeric_cols, key='box_y')
        color_col = st.sidebar.selectbox("Color by (optional):", [None] + all_cols, key='box_color')
        
    elif chart_type == "pie":
        values_col = st.sidebar.selectbox("Values:", numeric_cols, key='pie_values')
        names_col = st.sidebar.selectbox("Names:", all_cols, key='pie_names')
        
    elif chart_type in ["sunburst", "treemap"]:
        path_cols = st.sidebar.multiselect("Path (Hierarchy):", all_cols, key='path')
        values_col = st.sidebar.selectbox("Values:", numeric_cols, key='hierarchy_values')
        
    elif chart_type == "scatter_3d":
        x_axis = st.sidebar.selectbox("X-axis:", numeric_cols, key='3d_x')
        y_axis = st.sidebar.selectbox("Y-axis:", numeric_cols, key='3d_y', index=min(1, len(numeric_cols)-1))
        z_axis = st.sidebar.selectbox("Z-axis:", numeric_cols, key='3d_z', index=min(2, len(numeric_cols)-1))
        color_col = st.sidebar.selectbox("Color by (optional):", [None] + all_cols, key='3d_color')
        size_col = st.sidebar.selectbox("Size by (optional):", [None] + numeric_cols, key='3d_size')
        
    elif chart_type == "heatmap":
        heatmap_cols = st.sidebar.multiselect("Select Columns for Heatmap:", numeric_cols, 
                                             default=numeric_cols[:min(10, len(numeric_cols))], key='heatmap_cols')
        
    elif chart_type == "parallel":
        parallel_cols = st.sidebar.multiselect("Select Columns:", numeric_cols, 
                                              default=numeric_cols[:min(6, len(numeric_cols))], key='parallel_cols')
        color_col = st.sidebar.selectbox("Color by (optional):", [None] + all_cols, key='parallel_color')
        
    elif chart_type == "radar":
        radar_cols = st.sidebar.multiselect("Select Metrics:", numeric_cols, 
                                           default=numeric_cols[:min(5, len(numeric_cols))], key='radar_cols')
        category_col = st.sidebar.selectbox("Category (optional):", [None] + all_cols, key='radar_category')
        
    elif chart_type == "density_heatmap":
        x_axis = st.sidebar.selectbox("X-axis:", numeric_cols, key='density_x')
        y_axis = st.sidebar.selectbox("Y-axis:", numeric_cols, key='density_y', index=min(1, len(numeric_cols)-1))
        
    # Generate the selected chart
    st.subheader(f"📊 {selected_chart}")
    
    try:
        if chart_type == "scatter":
            fig = px.scatter(df_sample, x=x_axis, y=y_axis, color=color_col,
                           title=f'Scatter Plot: {x_axis} vs {y_axis}',
                           color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "line":
            fig = px.line(df_sample, x=x_axis, y=y_axis, color=color_col,
                         title=f'Line Chart: {y_axis} over {x_axis}',
                         color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "bar":
            if y_axis:
                fig = px.bar(df_sample, x=x_axis, y=y_axis, color=color_col,
                           title=f'Bar Chart: {y_axis} by {x_axis}',
                           color_discrete_sequence=colors if isinstance(colors, list) else None)
            else:
                # Count plot
                value_counts = df_sample[x_axis].value_counts().head(20)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f'Count Plot: {x_axis}',
                           color_discrete_sequence=colors if isinstance(colors, list) else None)
                fig.update_layout(xaxis_title=x_axis, yaxis_title='Count')
                
        elif chart_type == "histogram":
            fig = px.histogram(df_sample, x=x_axis, color=color_col, nbins=bins,
                             title=f'Histogram: {x_axis}',
                             color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "box":
            fig = px.box(df_sample, x=x_axis, y=y_axis, color=color_col,
                        title=f'Box Plot: {y_axis} by {x_axis}' if x_axis else f'Box Plot: {y_axis}',
                        color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "violin":
            fig = px.violin(df_sample, x=x_axis, y=y_axis, color=color_col,
                           title=f'Violin Plot: {y_axis} by {x_axis}' if x_axis else f'Violin Plot: {y_axis}',
                           color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "area":
            fig = px.area(df_sample, x=x_axis, y=y_axis, color=color_col,
                         title=f'Area Chart: {y_axis} over {x_axis}',
                         color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "pie":
            # Aggregate data if needed
            if df_sample[names_col].dtype == 'object':
                pie_data = df_sample.groupby(names_col)[values_col].sum().reset_index()
                fig = px.pie(pie_data, values=values_col, names=names_col,
                           title=f'Pie Chart: {values_col} by {names_col}',
                           color_discrete_sequence=colors if isinstance(colors, list) else None)
            else:
                fig = px.pie(df_sample, values=values_col, names=names_col,
                           title=f'Pie Chart: {values_col} by {names_col}',
                           color_discrete_sequence=colors if isinstance(colors, list) else None)
                
        elif chart_type == "sunburst":
            if len(path_cols) > 0:
                fig = px.sunburst(df_sample, path=path_cols, values=values_col,
                                title=f'Sunburst Chart: {" > ".join(path_cols)}',
                                color_discrete_sequence=colors if isinstance(colors, list) else None)
            else:
                st.warning("Please select at least one column for the path hierarchy.")
                return
                
        elif chart_type == "treemap":
            if len(path_cols) > 0:
                fig = px.treemap(df_sample, path=path_cols, values=values_col,
                               title=f'Treemap: {" > ".join(path_cols)}',
                               color_discrete_sequence=colors if isinstance(colors, list) else None)
            else:
                st.warning("Please select at least one column for the path hierarchy.")
                return
                
        elif chart_type == "scatter_3d":
            fig = px.scatter_3d(df_sample, x=x_axis, y=y_axis, z=z_axis,
                              color=color_col, size=size_col,
                              title=f'3D Scatter: {x_axis} vs {y_axis} vs {z_axis}',
                              color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "heatmap":
            if len(heatmap_cols) > 1:
                corr_matrix = df_sample[heatmap_cols].corr()
                fig = px.imshow(corr_matrix, title="Correlation Heatmap",
                              color_continuous_scale=colors[0] if isinstance(colors[0], str) else 'Viridis',
                              aspect="auto")
            else:
                st.warning("Please select at least 2 columns for the heatmap.")
                return
                
        elif chart_type == "correlation":
            if len(numeric_cols) > 1:
                corr_matrix = df_sample[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="Full Correlation Matrix",
                              color_continuous_scale=colors[0] if isinstance(colors[0], str) else 'Viridis',
                              aspect="auto")
            else:
                st.warning("Need at least 2 numeric columns for correlation matrix.")
                return
                
        elif chart_type == "distribution":
            col1, col2 = st.columns(2)
            
            # Select column for distribution
            dist_col = st.sidebar.selectbox("Select column:", numeric_cols, key='dist_col')
            
            with col1:
                fig1 = px.histogram(df_sample, x=dist_col, marginal="box",
                                  title=f'Distribution: {dist_col}',
                                  color_discrete_sequence=[colors[0] if isinstance(colors[0], str) else colors[0]])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Q-Q plot
                from scipy.stats import probplot
                qq_data = probplot(df_sample[dist_col].dropna(), dist="norm")
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                                        mode='markers', name='Sample Quantiles',
                                        marker=dict(color=colors[1] if isinstance(colors[1], str) else colors[1])))
                fig2.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0], 
                                        mode='lines', name='Theoretical Line',
                                        line=dict(color='red')))
                fig2.update_layout(title=f'Q-Q Plot: {dist_col}', 
                                 xaxis_title='Theoretical Quantiles',
                                 yaxis_title='Sample Quantiles')
                st.plotly_chart(fig2, use_container_width=True)
            return
            
        elif chart_type == "pair_plot":
            pair_cols = st.sidebar.multiselect("Select columns for pair plot:", numeric_cols, 
                                              default=numeric_cols[:min(4, len(numeric_cols))], key='pair_cols')
            
            if len(pair_cols) > 1:
                fig = px.scatter_matrix(df_sample, dimensions=pair_cols,
                                      title="Pair Plot Matrix",
                                      color_discrete_sequence=colors if isinstance(colors, list) else None)
            else:
                st.warning("Please select at least 2 columns for pair plot.")
                return
                
        elif chart_type == "parallel":
            if len(parallel_cols) > 1:
                fig = px.parallel_coordinates(df_sample, dimensions=parallel_cols, color=color_col,
                                            title="Parallel Coordinates Plot",
                                            color_continuous_scale=colors[0] if isinstance(colors[0], str) else 'Viridis')
            else:
                st.warning("Please select at least 2 columns for parallel coordinates.")
                return
                
        elif chart_type == "radar":
            if len(radar_cols) >= 3:
                if category_col and df_sample[category_col].nunique() <= 10:
                    # Multi-category radar chart
                    categories = df_sample[category_col].unique()[:5]
                    
                    fig = go.Figure()
                    
                    for i, category in enumerate(categories):
                        category_data = df_sample[df_sample[category_col] == category][radar_cols].mean()
                        
                        color_idx = i % len(colors) if isinstance(colors, list) else i
                        color = colors[color_idx] if isinstance(colors, list) else px.colors.qualitative.Plotly[i]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=category_data.values,
                            theta=radar_cols,
                            fill='toself',
                            name=str(category),
                            line_color=color
                        ))
                else:
                    # Single radar chart with mean values
                    mean_values = df_sample[radar_cols].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=mean_values.values,
                        theta=radar_cols,
                        fill='toself',
                        name='Mean Values',
                        line_color=colors[0] if isinstance(colors, list) else 'blue'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max([df_sample[col].max() for col in radar_cols])])
                    ),
                    showlegend=True,
                    title="Radar Chart"
                )
            else:
                st.warning("Please select at least 3 columns for radar chart.")
                return
                
        elif chart_type == "bubble":
            fig = px.scatter(df_sample, x=x_axis, y=y_axis, size=size_col, color=color_col,
                           title=f'Bubble Chart: {x_axis} vs {y_axis} (Size: {size_col})',
                           color_discrete_sequence=colors if isinstance(colors, list) else None)
            
        elif chart_type == "density_heatmap":
            fig = px.density_heatmap(df_sample, x=x_axis, y=y_axis,
                                   title=f'Density Heatmap: {x_axis} vs {y_axis}',
                                   color_continuous_scale=colors[0] if isinstance(colors[0], str) else 'Viridis')
            
        elif chart_type == "stats_summary":
            if numeric_cols:
                summary_stats = df_sample[numeric_cols].describe()
                
                st.subheader("Statistical Summary")
                st.dataframe(summary_stats, use_container_width=True)
                
                # Additional statistics
                additional_stats = pd.DataFrame({
                    'skewness': df_sample[numeric_cols].skew(),
                    'kurtosis': df_sample[numeric_cols].kurtosis(),
                    'variance': df_sample[numeric_cols].var(),
                    'coefficient_of_variation': (df_sample[numeric_cols].std() / df_sample[numeric_cols].mean()) * 100
                })
                
                st.subheader("Advanced Statistics")
                st.dataframe(additional_stats.T, use_container_width=True)
                
                # Missing values summary
                missing_data = df_sample.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if not missing_data.empty:
                    fig = px.bar(x=missing_data.index, y=missing_data.values,
                               title="Missing Values by Column",
                               color_discrete_sequence=[colors[0] if isinstance(colors[0], str) else colors[0]])
                    fig.update_layout(xaxis_title="Columns", yaxis_title="Missing Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing values found in the dataset!")
                
                return
            else:
                st.warning("No numeric columns found for statistical summary.")
                return
        
        # Display the chart
        if 'fig' in locals():
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Chart insights
            st.subheader("📝 Chart Insights")
            
            if chart_type == "scatter":
                if color_col and x_axis in numeric_cols and y_axis in numeric_cols:
                    corr = df_sample[x_axis].corr(df_sample[y_axis])
                    st.write(f"**Correlation between {x_axis} and {y_axis}:** {corr:.3f}")
                    
            elif chart_type == "histogram":
                mean_val = df_sample[x_axis].mean()
                median_val = df_sample[x_axis].median()
                std_val = df_sample[x_axis].std()
                st.write(f"**Mean:** {mean_val:.3f} | **Median:** {median_val:.3f} | **Std Dev:** {std_val:.3f}")
                
            elif chart_type in ["box", "violin"]:
                if x_axis:
                    group_stats = df_sample.groupby(x_axis)[y_axis].agg(['mean', 'median', 'std']).round(3)
                    st.dataframe(group_stats, use_container_width=True)
                    
            elif chart_type == "heatmap" and len(heatmap_cols) > 1:
                # Find strongest correlations
                corr_matrix = df_sample[heatmap_cols].corr()
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_corr.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)
                    st.write("**Strong Correlations (|r| > 0.5):**")
                    st.dataframe(strong_corr_df, use_container_width=True)
            
            # Download chart data
            if chart_type not in ["stats_summary", "distribution"]:
                chart_data = df_sample.copy()
                csv_data = chart_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Chart Data",
                    data=csv_data,
                    file_name=f"{selected_chart.lower().replace(' ', '_')}_data.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error creating {selected_chart}: {str(e)}")
        st.info("Please check your column selections and try again.")

def create_advanced_ml_models():
    """Create comprehensive machine learning model library with single-threading"""
    
    classification_models = {
        # Tree-based models
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'category': 'Ensemble'
        },
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=42, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'category': 'Ensemble'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'category': 'Ensemble'
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'category': 'Boosting'
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100]
            },
            'category': 'Boosting'
        },
        'CatBoost': {
            'model': CatBoostClassifier(random_state=42, silent=True, thread_count=1),  # 🔧 FIX: Added thread_count=1
            'params': {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7]
            },
            'category': 'Boosting'
        },
        
        # Linear models
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'category': 'Linear'
        },
        'Linear SVM': {
            'model': LinearSVC(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2']
            },
            'category': 'SVM'
        },
        'SVM (RBF)': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            },
            'category': 'SVM'
        },
        
        # Instance-based
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'category': 'Instance-based'
        },
        
        # Naive Bayes
        'Gaussian Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'category': 'Probabilistic'
        },
        
        # Neural Networks
        'Multi-layer Perceptron': {
            'model': MLPClassifier(random_state=42, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'category': 'Neural Network'
        },
        
        # Discriminant Analysis
        'Linear Discriminant Analysis': {
            'model': LinearDiscriminantAnalysis(),
            'params': {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
            },
            'category': 'Discriminant'
        },
        'Quadratic Discriminant Analysis': {
            'model': QuadraticDiscriminantAnalysis(),
            'params': {
                'reg_param': [0.0, 0.1, 0.5, 0.9]
            },
            'category': 'Discriminant'
        }
    }
    
    regression_models = {
        # Tree-based models
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'category': 'Ensemble'
        },
        'Extra Trees': {
            'model': ExtraTreesRegressor(random_state=42, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'category': 'Ensemble'
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'category': 'Ensemble'
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'category': 'Boosting'
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42, verbose=-1, n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100]
            },
            'category': 'Boosting'
        },
        'CatBoost': {
            'model': CatBoostRegressor(random_state=42, silent=True, thread_count=1),  # 🔧 FIX: Added thread_count=1
            'params': {
                'iterations': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7]
            },
            'category': 'Boosting'
        },
        
        # Linear models
        'Linear Regression': {
            'model': LinearRegression(n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            'category': 'Linear'
        },
        'Ridge Regression': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.1, 1, 10, 100, 1000],
                'fit_intercept': [True, False]
            },
            'category': 'Linear'
        },
        'Lasso Regression': {
            'model': Lasso(random_state=42),
            'params': {
                'alpha': [0.1, 1, 10, 100, 1000],
                'fit_intercept': [True, False]
            },
            'category': 'Linear'
        },
        'Elastic Net': {
            'model': ElasticNet(random_state=42),
            'params': {
                'alpha': [0.1, 1, 10, 100],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'category': 'Linear'
        },
        
        # SVM
        'Linear SVR': {
            'model': LinearSVR(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2]
            },
            'category': 'SVM'
        },
        'SVR (RBF)': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['rbf', 'poly']
            },
            'category': 'SVM'
        },
        
        # Instance-based
        'K-Nearest Neighbors': {
            'model': KNeighborsRegressor(n_jobs=1),  # 🔧 FIX: Added n_jobs=1
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'category': 'Instance-based'
        },
        
        # Neural Networks
        'Multi-layer Perceptron': {
            'model': MLPRegressor(random_state=42, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'category': 'Neural Network'
        }
    }
    
    return classification_models, regression_models

def create_deep_learning_models(input_shape, task_type='classification', complexity='medium'):
    """Create advanced deep learning models with different architectures"""
    
    models = {}
    
    # Simple Feed-Forward Neural Network
    def create_ffnn(complexity_level):
        model = Sequential()
        model.add(Dense(input_shape[0], input_shape=input_shape))
        
        if complexity_level == 'simple':
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
        elif complexity_level == 'medium':
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
        else:  # complex
            model.add(Dense(256, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
        
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])
        
        return model
    
    # Convolutional Neural Network (1D)
    def create_cnn1d(complexity_level):
        model = Sequential()
        model.add(tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape))
        
        if complexity_level == 'simple':
            model.add(Conv1D(32, kernel_size=3, activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(32, activation='relu'))
        elif complexity_level == 'medium':
            model.add(Conv1D(64, kernel_size=3, activation='relu'))
            model.add(Conv1D(32, kernel_size=3, activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
        else:  # complex
            model.add(Conv1D(128, kernel_size=5, activation='relu'))
            model.add(Conv1D(64, kernel_size=3, activation='relu'))
            model.add(Conv1D(32, kernel_size=3, activation='relu'))
            model.add(GlobalMaxPooling1D())
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
        
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])
        
        return model
    
    # LSTM Network
    def create_lstm(complexity_level):
        model = Sequential()
        model.add(tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape))
        
        if complexity_level == 'simple':
            model.add(LSTM(32, return_sequences=False))
            model.add(Dense(16, activation='relu'))
        elif complexity_level == 'medium':
            model.add(LSTM(64, return_sequences=True))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
        else:  # complex
            model.add(LSTM(128, return_sequences=True))
            model.add(LSTM(64, return_sequences=True))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(32, activation='relu'))
        
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])
        
        return model
    
    # Bidirectional LSTM
    def create_bidirectional_lstm(complexity_level):
        model = Sequential()
        model.add(tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape))
        
        if complexity_level == 'simple':
            model.add(Bidirectional(LSTM(32)))
            model.add(Dense(16, activation='relu'))
        elif complexity_level == 'medium':
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
        else:  # complex
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])
        
        return model
    
    # GRU Network
    def create_gru(complexity_level):
        model = Sequential()
        model.add(tf.keras.layers.Reshape((input_shape[0], 1), input_shape=input_shape))
        
        if complexity_level == 'simple':
            model.add(GRU(32))
            model.add(Dense(16, activation='relu'))
        elif complexity_level == 'medium':
            model.add(GRU(64, return_sequences=True))
            model.add(GRU(32))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))
        else:  # complex
            model.add(GRU(128, return_sequences=True))
            model.add(GRU(64, return_sequences=True))
            model.add(GRU(32))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])
        
        return model
    
    # Create models based on complexity
    models[f'FFNN_{complexity}'] = create_ffnn(complexity)
    models[f'CNN1D_{complexity}'] = create_cnn1d(complexity)
    models[f'LSTM_{complexity}'] = create_lstm(complexity)
    models[f'BiLSTM_{complexity}'] = create_bidirectional_lstm(complexity)
    models[f'GRU_{complexity}'] = create_gru(complexity)
    
    return models

def intelligent_model_recommendation_system(X_train, y_train, X_test, y_test, task_type='classification'):
    """Advanced model recommendation system with comprehensive evaluation"""
    
    st.markdown('<h2 class="sub-header">🤖 Intelligent Model Recommendation System</h2>', unsafe_allow_html=True)
    
    # Get model libraries
    if task_type == 'classification':
        ml_models, _ = create_advanced_ml_models()
        scoring_metric = 'accuracy'
    else:
        _, ml_models = create_advanced_ml_models()
        scoring_metric = 'neg_mean_squared_error'
    
    # Model evaluation results
    model_results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_models = len(ml_models)
    
    st.subheader("🔄 Machine Learning Model Evaluation")
    
    # Evaluate ML models
    successful_models = 0
    
    for idx, (model_name, model_config) in enumerate(ml_models.items()):
        status_text.text(f'Evaluating {model_name}...')
        progress_bar.progress((idx + 1) / total_models)
        
        try:
            start_time = time.time()
            
            # Get the model instance
            model = model_config['model']
            
            # 🔧 FIX: Ensure single-threading
            if hasattr(model, 'n_jobs'):
                model.set_params(n_jobs=1)
            if hasattr(model, 'thread_count'):  # For CatBoost
                model.set_params(thread_count=1)
            
            # 🔧 FIX: Use simple train-test split instead of cross-validation to avoid multiprocessing
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Fit model
            model.fit(X_tr, y_tr)
            
            if task_type == 'classification':
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate validation score (simpler than cross-validation)
                val_score = accuracy_score(y_val, model.predict(X_val))
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                model_results.append({
                    'Model': model_name,
                    'Category': model_config['category'],
                    'CV_Mean': val_score,  # Use validation score instead of CV
                    'CV_Std': 0.0,  # Not available with simple validation
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'ROC_AUC': roc_auc,
                    'Training_Time': time.time() - start_time,
                    'Task_Type': 'Classification'
                })
                
            else:  # regression
                y_pred = model.predict(X_test)
                
                # Calculate validation score
                val_score = r2_score(y_val, model.predict(X_val))
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_results.append({
                    'Model': model_name,
                    'Category': model_config['category'],
                    'CV_Mean': val_score,  # Use validation score
                    'CV_Std': 0.0,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2_Score': r2,
                    'Training_Time': time.time() - start_time,
                    'Task_Type': 'Regression'
                })
            
            successful_models += 1
            st.success(f"✅ {model_name} evaluated successfully!")
                
        except Exception as e:
            st.warning(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if successful_models == 0:
        st.error("❌ No models could be evaluated successfully. Please check your data and try again.")
        return None, None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(model_results)
    
    # Display results
    st.subheader("📊 Model Performance Comparison")
    
    if task_type == 'classification':
        # Sort by accuracy for classification
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        # Display top performers
        st.dataframe(results_df[['Model', 'Category', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'Training_Time']].round(4), 
                    use_container_width=True)
        
        # Best model identification
        best_model = results_df.iloc[0]
        
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Best Model: {best_model['Model']}</h3>
            <p><strong>Category:</strong> {best_model['Category']}</p>
            <p><strong>Accuracy:</strong> {best_model['Accuracy']:.4f}</p>
            <p><strong>F1 Score:</strong> {best_model['F1_Score']:.4f}</p>
            <p><strong>ROC AUC:</strong> {best_model['ROC_AUC']:.4f if best_model['ROC_AUC'] else 'N/A'}</p>
            <p><strong>Training Time:</strong> {best_model['Training_Time']:.2f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df.head(10), x='Accuracy', y='Model', 
                        title='Top 10 Models by Accuracy',
                        color='Category',
                        orientation='h')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(results_df, x='Training_Time', y='Accuracy', 
                           color='Category', size='F1_Score',
                           title='Accuracy vs Training Time',
                           hover_data=['Model'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # regression
        # Sort by R2 score for regression
        results_df = results_df.sort_values('R2_Score', ascending=False)
        
        # Display top performers
        st.dataframe(results_df[['Model', 'Category', 'R2_Score', 'RMSE', 'MAE', 'Training_Time']].round(4), 
                    use_container_width=True)
        
        # Best model identification
        best_model = results_df.iloc[0]
        
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Best Model: {best_model['Model']}</h3>
            <p><strong>Category:</strong> {best_model['Category']}</p>
            <p><strong>R² Score:</strong> {best_model['R2_Score']:.4f}</p>
            <p><strong>RMSE:</strong> {best_model['RMSE']:.4f}</p>
            <p><strong>MAE:</strong> {best_model['MAE']:.4f}</p>
            <p><strong>Training Time:</strong> {best_model['Training_Time']:.2f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df.head(10), x='R2_Score', y='Model', 
                        title='Top 10 Models by R² Score',
                        color='Category',
                        orientation='h')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(results_df, x='Training_Time', y='R2_Score', 
                           color='Category', size='MAE',
                           title='R² Score vs Training Time',
                           hover_data=['Model'])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Deep Learning Model Evaluation
    if st.checkbox("🧠 Include Deep Learning Models", value=False):
        st.subheader("🔄 Deep Learning Model Evaluation")
        
        # Create deep learning models
        input_shape = (X_train.shape[1],)
        complexities = ['simple', 'medium', 'complex']
        
        dl_results = []
        
        for complexity in complexities:
            dl_models = create_deep_learning_models(input_shape, task_type, complexity)
            
            for model_name, model in dl_models.items():
                status_text.text(f'Evaluating {model_name}...')
                
                try:
                    start_time = time.time()
                    
                    # Train deep learning model
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
                    ]
                    
                    history = model.fit(X_train, y_train, 
                                      epochs=50, 
                                      batch_size=32,
                                      validation_split=0.2,
                                      callbacks=callbacks,
                                      verbose=0)
                    
                    # Evaluate model
                    if task_type == 'classification':
                        y_pred_proba = model.predict(X_test)
                        y_pred = (y_pred_proba > 0.5).astype(int).ravel()
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                        
                        dl_results.append({
                            'Model': model_name,
                            'Category': 'Deep Learning',
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1_Score': f1,
                            'ROC_AUC': roc_auc,
                            'Training_Time': time.time() - start_time,
                            'Final_Loss': history.history['loss'][-1],
                            'Final_Val_Loss': history.history['val_loss'][-1],
                            'Task_Type': 'Classification'
                        })
                        
                    else:  # regression
                        y_pred = model.predict(X_test).ravel()
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        dl_results.append({
                            'Model': model_name,
                            'Category': 'Deep Learning',
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAE': mae,
                            'R2_Score': r2,
                            'Training_Time': time.time() - start_time,
                            'Final_Loss': history.history['loss'][-1],
                            'Final_Val_Loss': history.history['val_loss'][-1],
                            'Task_Type': 'Regression'
                        })
                        
                except Exception as e:
                    st.warning(f"Error evaluating {model_name}: {str(e)}")
                    continue
        
        # Add deep learning results to main results
        if dl_results:
            dl_results_df = pd.DataFrame(dl_results)
            
            if task_type == 'classification':
                combined_results = pd.concat([results_df, dl_results_df], ignore_index=True)
                combined_results = combined_results.sort_values('Accuracy', ascending=False)
            else:
                combined_results = pd.concat([results_df, dl_results_df], ignore_index=True)
                combined_results = combined_results.sort_values('R2_Score', ascending=False)
            
            st.subheader("🤖 Combined ML + DL Performance")
            
            if task_type == 'classification':
                st.dataframe(combined_results[['Model', 'Category', 'Accuracy', 'F1_Score', 'ROC_AUC', 'Training_Time']].round(4), 
                           use_container_width=True)
                
                # Update best model if DL performs better
                new_best = combined_results.iloc[0]
                if new_best['Category'] == 'Deep Learning':
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>🧠 New Best Model (Deep Learning): {new_best['Model']}</h3>
                        <p><strong>Accuracy:</strong> {new_best['Accuracy']:.4f}</p>
                        <p><strong>F1 Score:</strong> {new_best['F1_Score']:.4f}</p>
                        <p><strong>ROC AUC:</strong> {new_best['ROC_AUC']:.4f}</p>
                        <p><strong>Training Time:</strong> {new_best['Training_Time']:.2f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.dataframe(combined_results[['Model', 'Category', 'R2_Score', 'RMSE', 'MAE', 'Training_Time']].round(4), 
                           use_container_width=True)
                
                new_best = combined_results.iloc[0]
                if new_best['Category'] == 'Deep Learning':
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>🧠 New Best Model (Deep Learning): {new_best['Model']}</h3>
                        <p><strong>R² Score:</strong> {new_best['R2_Score']:.4f}</p>
                        <p><strong>RMSE:</strong> {new_best['RMSE']:.4f}</p>
                        <p><strong>MAE:</strong> {new_best['MAE']:.4f}</p>
                        <p><strong>Training Time:</strong> {new_best['Training_Time']:.2f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            results_df = combined_results
    
    # Model interpretability and recommendations
    st.subheader("🔍 Model Analysis and Recommendations")
    
    # Category-wise performance analysis
    category_performance = results_df.groupby('Category').agg({
        'Accuracy' if task_type == 'classification' else 'R2_Score': ['mean', 'std'],
        'Training_Time': ['mean', 'std']
    }).round(4)
    
    st.write("📊 **Category-wise Performance Summary:**")
    st.dataframe(category_performance, use_container_width=True)
    
    # Recommendations based on different criteria
    st.subheader("💡 Smart Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if task_type == 'classification':
            best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
        else:
            best_accuracy = results_df.loc[results_df['R2_Score'].idxmax()]
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Best Performance</h4>
            <p><strong>{best_accuracy['Model']}</strong></p>
            <p>{best_accuracy['Category']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fastest_model = results_df.loc[results_df['Training_Time'].idxmin()]
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Fastest Training</h4>
            <p><strong>{fastest_model['Model']}</strong></p>
            <p>{fastest_model['Training_Time']:.2f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Balanced recommendation (performance vs speed)
        if task_type == 'classification':
            results_df['efficiency_score'] = (results_df['Accuracy'] / results_df['Training_Time']) * 100
            best_balanced = results_df.loc[results_df['efficiency_score'].idxmax()]
        else:
            results_df['efficiency_score'] = (results_df['R2_Score'] / results_df['Training_Time']) * 100
            best_balanced = results_df.loc[results_df['efficiency_score'].idxmax()]
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚖️ Best Balanced</h4>
            <p><strong>{best_balanced['Model']}</strong></p>
            <p>Efficiency: {best_balanced['efficiency_score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    return results_df, best_model

def create_model_deployment_code(best_model_name, features, target, task_type):
    """Generate deployment-ready code for the best model"""
    
    st.subheader("📦 Model Deployment Code Generator")
    
    deployment_code = f"""
# Model Deployment Code - {best_model_name}
# Generated by Advanced AutoML Dashboard

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

class ModelDeployment:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = {features}
        self.target_name = '{target}'
        self.task_type = '{task_type}'
        
    def load_model(self, model_path, scaler_path=None):
        '''Load the trained model and scaler'''
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        print("Model loaded successfully!")
        
    def preprocess_input(self, input_data):
        '''Preprocess input data for prediction'''
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {{missing_features}}")
        
        # Select and order features
        input_data = input_data[self.feature_names]
        
        # Scale features if scaler is available
        if self.scaler:
            input_data = self.scaler.transform(input_data)
        
        return input_data
    
    def predict(self, input_data):
        '''Make predictions on new data'''
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        processed_data = self.preprocess_input(input_data)
        
        if self.task_type == 'classification':
            predictions = self.model.predict(processed_data)
            probabilities = None
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
            
            return {{
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None
            }}
        else:
            predictions = self.model.predict(processed_data)
            return {{
                'predictions': predictions.tolist()
            }}
    
    def batch_predict(self, csv_file_path, output_path):
        '''Perform batch predictions on a CSV file'''
        data = pd.read_csv(csv_file_path)
        results = self.predict(data)
        
        # Add predictions to original data
        data['predictions'] = results['predictions']
        if results.get('probabilities'):
            data['probabilities'] = results['probabilities']
        
        data.to_csv(output_path, index=False)
        print(f"Batch predictions saved to {{output_path}}")

# Example usage:
if __name__ == "__main__":
    # Initialize deployment
    deployment = ModelDeployment()
    
    # Load your trained model
    # deployment.load_model('model.pkl', 'scaler.pkl')
    
    # Make prediction on new data
    sample_input = {{
        # Add sample feature values here
        {", ".join([f"'{feature}': 0.0" for feature in features[:5]])}
    }}
    
    # result = deployment.predict(sample_input)
    # print("Prediction:", result)
    
    # For batch predictions:
    # deployment.batch_predict('new_data.csv', 'predictions.csv')
"""
    
    st.code(deployment_code, language='python')
    
    # Download button for the code
    st.download_button(
        label="📥 Download Deployment Code",
        data=deployment_code,
        file_name=f"{best_model_name.lower().replace(' ', '_')}_deployment.py",
        mime="text/plain"
    )

def main():
    """Main application function"""
    
    st.markdown('<h1 class="main-header">🚀 Advanced AutoML & Visualization Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Complete Data Science Pipeline: From Raw Data to Production-Ready Models
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
    
    # Sidebar configuration
    st.sidebar.header('⚙️ Configuration')
    
    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset (CSV, XLSX, XLS)", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to begin the analysis"
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
            
            # Application mode selection - Use cleaned data if available, otherwise use original
            current_data = st.session_state.processed_data if st.session_state.processed_data is not None else df
            
            st.sidebar.subheader('🎯 Application Mode')
            app_mode = st.sidebar.radio(
                "Choose your analysis type:",
                ["📊 Data Visualization", "🤖 Model Recommendation", "🔍 Advanced Analytics"]
            )
            
            if app_mode == "📊 Data Visualization":
                # Check dataset size for visualization
                if len(current_data) > 100000:
                    st.warning("⚠️ Large dataset detected. Visualization will use a sample for performance.")
                
                create_advanced_visualizations(current_data)
            
            elif app_mode == "🤖 Model Recommendation":
                st.markdown('<h2 class="sub-header">🎯 Machine Learning Model Selection</h2>', unsafe_allow_html=True)
                
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
                    default=available_features[:min(20, len(available_features))]  # Limit default selection
                )
                
                # Advanced options
                with st.expander("⚙️ Advanced Model Options", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        test_size = st.slider("Test set size", 0.1, 0.4, 0.2)
                        random_state = st.number_input("Random state", value=42)
                    
                    with col2:
                        scale_features = st.checkbox("Scale features", value=True)
                        feature_selection = st.checkbox("Apply feature selection", value=False)
                
                # Model training and evaluation
                if st.button("🚀 Start Model Training & Evaluation", type="primary"):
                    if len(selected_features) == 0:
                        st.error("❌ Please select at least one feature column.")
                    else:
                        with st.spinner('🔄 Training and evaluating models...'):
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
                            if scale_features:
                                scaler = StandardScaler()
                                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                            
                            # Feature selection
                            if feature_selection:
                                if task_type == 'classification':
                                    selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
                                else:
                                    selector = SelectKBest(f_regression, k=min(10, X.shape[1]))
                                
                                X = pd.DataFrame(selector.fit_transform(X, y), 
                                               columns=X.columns[selector.get_support()], 
                                               index=X.index)
                                
                                st.info(f"🔍 Selected {X.shape[1]} most important features")
                            
                            # Train-test split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state, 
                                stratify=y if task_type == 'classification' and len(y.unique()) > 1 else None
                            )
                            
                            st.info(f"📊 Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
                            
                            # Model evaluation
                            st.session_state.model_results, st.session_state.best_model = intelligent_model_recommendation_system(
                                X_train, y_train, X_test, y_test, task_type
                            )
                            
                            # Generate deployment code
                            if st.session_state.best_model is not None:
                                create_model_deployment_code(
                                    st.session_state.best_model['Model'], 
                                    selected_features, 
                                    target_column, 
                                    task_type
                                )
            
            elif app_mode == "🔍 Advanced Analytics":
                st.markdown('<h2 class="sub-header">🔬 Advanced Analytics Suite</h2>', unsafe_allow_html=True)
                
                analytics_options = st.multiselect(
                    "Select advanced analytics to perform:",
                    [
                        "Principal Component Analysis (PCA)",
                        "Clustering Analysis", 
                        "Feature Importance Analysis",
                        "Statistical Testing",
                        "Anomaly Detection",
                        "Time Series Decomposition"
                    ]
                )
                
                numeric_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
                
                if "Principal Component Analysis (PCA)" in analytics_options:
                    st.subheader("🔍 Principal Component Analysis")
                    
                    if len(numeric_columns) >= 2:
                        n_components = st.slider("Number of components:", 2, min(10, len(numeric_columns)), 3)
                        
                        # Perform PCA
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(current_data[numeric_columns])
                        
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(scaled_data)
                        
                        # Create PCA DataFrame
                        pca_df = pd.DataFrame(
                            data=pca_result,
                            columns=[f'PC{i+1}' for i in range(n_components)]
                        )
                        
                        # Explained variance
                        explained_var = pca.explained_variance_ratio_
                        cumulative_var = np.cumsum(explained_var)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Explained variance plot
                            fig = px.bar(x=range(1, n_components+1), y=explained_var,
                                       title="Explained Variance by Component",
                                       labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Cumulative explained variance
                            fig = px.line(x=range(1, n_components+1), y=cumulative_var,
                                        title="Cumulative Explained Variance",
                                        labels={'x': 'Number of Components', 'y': 'Cumulative Variance'})
                            fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                                        annotation_text="95% Variance")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 2D PCA visualization
                        if n_components >= 2:
                            fig = px.scatter(pca_df, x='PC1', y='PC2', 
                                           title="PCA: First Two Components")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Component loadings
                        loadings = pd.DataFrame(
                            pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=numeric_columns
                        )
                        
                        st.subheader("Component Loadings")
                        st.dataframe(loadings.round(3), use_container_width=True)
                    
                    else:
                        st.warning("❌ Need at least 2 numeric columns for PCA")
                
                if "Clustering Analysis" in analytics_options:
                    st.subheader("🎯 Clustering Analysis")
                    
                    if len(numeric_columns) >= 2:
                        clustering_method = st.selectbox("Select clustering method:", 
                                                       ["K-Means", "DBSCAN", "Agglomerative"])
                        
                        # Prepare data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(current_data[numeric_columns])
                        
                        if clustering_method == "K-Means":
                            n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                            
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                            cluster_labels = kmeans.fit_predict(scaled_data)
                            
                            # Add clusters to data
                            plot_data = current_data[numeric_columns].copy()
                            plot_data['Cluster'] = cluster_labels
                            
                            # Visualize clusters
                            if len(numeric_columns) >= 2:
                                fig = px.scatter(plot_data, x=numeric_columns[0], y=numeric_columns[1], 
                                               color='Cluster', title=f"K-Means Clustering (k={n_clusters})")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Cluster summary
                            cluster_summary = plot_data.groupby('Cluster')[numeric_columns].mean()
                            st.subheader("Cluster Centers")
                            st.dataframe(cluster_summary.round(3), use_container_width=True)
                        
                        elif clustering_method == "DBSCAN":
                            eps = st.slider("Epsilon (neighborhood distance):", 0.1, 2.0, 0.5)
                            min_samples = st.slider("Minimum samples:", 2, 20, 5)
                            
                            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                            cluster_labels = dbscan.fit_predict(scaled_data)
                            
                            # Add clusters to data
                            plot_data = current_data[numeric_columns].copy()
                            plot_data['Cluster'] = cluster_labels
                            
                            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                            n_noise = list(cluster_labels).count(-1)
                            
                            st.info(f"🎯 Found {n_clusters} clusters and {n_noise} noise points")
                            
                            # Visualize clusters
                            if len(numeric_columns) >= 2:
                                fig = px.scatter(plot_data, x=numeric_columns[0], y=numeric_columns[1], 
                                               color='Cluster', title="DBSCAN Clustering")
                                st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.warning("❌ Need at least 2 numeric columns for clustering")
                
                if "Feature Importance Analysis" in analytics_options:
                    st.subheader("🔬 Feature Importance Analysis")
                    
                    if len(numeric_columns) > 1:
                        target_for_importance = st.selectbox("Select target for importance:", numeric_columns)
                        
                        X = current_data[numeric_columns].drop(columns=[target_for_importance])
                        y = current_data[target_for_importance]
                        
                        # Random Forest for feature importance
                        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
                        rf.fit(X, y)
                        
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': rf.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Feature importance plot
                        fig = px.bar(importance_df.head(15), x='Importance', y='Feature',
                                   title="Feature Importance (Random Forest)",
                                   orientation='h')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation with target
                        correlations = X.corrwith(y).abs().sort_values(ascending=False)
                        corr_df = pd.DataFrame({
                            'Feature': correlations.index,
                            'Correlation': correlations.values
                        })
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Top Feature Importances")
                            st.dataframe(importance_df.head(10), use_container_width=True)
                        
                        with col2:
                            st.subheader("Top Correlations with Target")
                            st.dataframe(corr_df.head(10), use_container_width=True)
                    
                    else:
                        st.warning("❌ Need at least 2 numeric columns for feature importance analysis")
                
                if "Statistical Testing" in analytics_options:
                    st.subheader("📊 Statistical Testing Suite")
                    
                    if len(numeric_columns) >= 2:
                        test_type = st.selectbox("Select statistical test:", 
                                               ["Correlation Test", "T-Test", "ANOVA", "Chi-Square Test"])
                        
                        if test_type == "Correlation Test":
                            col1_test = st.selectbox("Select first variable:", numeric_columns, key="corr_var1")
                            col2_test = st.selectbox("Select second variable:", numeric_columns, key="corr_var2")
                            
                            if col1_test != col2_test:
                                # Pearson correlation
                                pearson_corr, pearson_p = stats.pearsonr(
                                    current_data[col1_test].dropna(),
                                    current_data[col2_test].dropna()
                                )
                                
                                # Spearman correlation
                                spearman_corr, spearman_p = stats.spearmanr(
                                    current_data[col1_test].dropna(),
                                    current_data[col2_test].dropna()
                                )
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Pearson Correlation", f"{pearson_corr:.4f}")
                                    st.metric("P-value", f"{pearson_p:.4f}")
                                    st.write("✅ Significant" if pearson_p < 0.05 else "❌ Not Significant")
                                
                                with col2:
                                    st.metric("Spearman Correlation", f"{spearman_corr:.4f}")
                                    st.metric("P-value", f"{spearman_p:.4f}")
                                    st.write("✅ Significant" if spearman_p < 0.05 else "❌ Not Significant")
                                
                                # Scatter plot with regression line
                                fig = px.scatter(current_data, 
                                               x=col1_test, y=col2_test,
                                               trendline="ols",
                                               title=f"Correlation: {col1_test} vs {col2_test}")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif test_type == "T-Test":
                            test_var = st.selectbox("Select test variable:", numeric_columns, key="ttest_var")
                            group_var = st.selectbox("Select grouping variable:", 
                                                    current_data.columns.tolist(), 
                                                    key="ttest_group")
                            
                            unique_groups = current_data[group_var].unique()
                            if len(unique_groups) == 2:
                                group1_data = current_data[
                                    current_data[group_var] == unique_groups[0]
                                ][test_var].dropna()
                                group2_data = current_data[
                                    current_data[group_var] == unique_groups[1]
                                ][test_var].dropna()
                                
                                # Independent t-test
                                t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("T-Statistic", f"{t_stat:.4f}")
                                with col2:
                                    st.metric("P-Value", f"{p_value:.4f}")
                                with col3:
                                    st.write("✅ Significant Difference" if p_value < 0.05 else "❌ No Significant Difference")
                                
                                # Box plot comparison
                                fig = px.box(current_data, x=group_var, y=test_var,
                                           title=f"T-Test: {test_var} by {group_var}")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                st.warning("❌ T-test requires exactly 2 groups. Selected variable has " + str(len(unique_groups)) + " groups.")
                
                if "Anomaly Detection" in analytics_options:
                    st.subheader("🚨 Anomaly Detection")
                    
                    if len(numeric_columns) >= 1:
                        from sklearn.ensemble import IsolationForest
                        from sklearn.neighbors import LocalOutlierFactor
                        
                        detection_method = st.selectbox("Select anomaly detection method:", 
                                                      ["Isolation Forest", "Local Outlier Factor", "Statistical (Z-Score)"])
                        
                        # Prepare data
                        anomaly_data = current_data[numeric_columns].copy()
                        
                        if detection_method == "Isolation Forest":
                            contamination = st.slider("Contamination rate:", 0.01, 0.5, 0.1)
                            
                            iso_forest = IsolationForest(contamination=contamination, random_state=42)
                            anomaly_labels = iso_forest.fit_predict(anomaly_data)
                            
                            anomaly_data['Anomaly'] = anomaly_labels
                            anomaly_data['Anomaly'] = anomaly_data['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
                            
                            n_anomalies = (anomaly_labels == -1).sum()
                            st.info(f"🚨 Detected {n_anomalies} anomalies ({n_anomalies/len(anomaly_data)*100:.2f}%)")
                            
                        elif detection_method == "Local Outlier Factor":
                            n_neighbors = st.slider("Number of neighbors:", 5, 50, 20)
                            
                            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
                            anomaly_labels = lof.fit_predict(anomaly_data)
                            
                            anomaly_data['Anomaly'] = anomaly_labels
                            anomaly_data['Anomaly'] = anomaly_data['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
                            
                            n_anomalies = (anomaly_labels == -1).sum()
                            st.info(f"🚨 Detected {n_anomalies} anomalies ({n_anomalies/len(anomaly_data)*100:.2f}%)")
                        
                        elif detection_method == "Statistical (Z-Score)":
                            z_threshold = st.slider("Z-Score threshold:", 2.0, 4.0, 3.0)
                            
                            # Calculate Z-scores for each numeric column
                            z_scores = np.abs(stats.zscore(anomaly_data))
                            anomaly_mask = (z_scores > z_threshold).any(axis=1)
                            
                            anomaly_data['Anomaly'] = anomaly_mask.map({True: 'Anomaly', False: 'Normal'})
                            
                            n_anomalies = anomaly_mask.sum()
                            st.info(f"🚨 Detected {n_anomalies} anomalies ({n_anomalies/len(anomaly_data)*100:.2f}%)")
                        
                        # Visualize anomalies
                        if len(numeric_columns) >= 2:
                            fig = px.scatter(anomaly_data, x=numeric_columns[0], y=numeric_columns[1], 
                                           color='Anomaly', 
                                           title=f"Anomaly Detection: {detection_method}",
                                           color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show anomaly summary
                        anomaly_summary = anomaly_data.groupby('Anomaly')[numeric_columns].describe()
                        st.subheader("Anomaly Summary Statistics")
                        st.dataframe(anomaly_summary, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is properly formatted and try again.")
    
    else:
        # Landing page content
        st.markdown("""
        ## 🌟 Features
        
        ### 📊 **Advanced Data Visualization**
        - 20+ interactive chart types including 3D visualizations
        - Real-time data profiling and statistical analysis
        - Correlation analysis and distribution testing
        - Time series analysis and seasonal decomposition
        
        ### 🤖 **Intelligent Model Recommendation**
        - 25+ machine learning algorithms with hyperparameter tuning
        - Deep learning models with different architectures
        - Automated model selection based on performance metrics
        - Cross-validation and statistical evaluation
        
        ### 🔬 **Advanced Analytics Suite**
        - Principal Component Analysis (PCA)
        - Clustering analysis (K-Means, DBSCAN, Agglomerative)
        - Feature importance and selection
        - Statistical testing (T-tests, ANOVA, Correlation)
        - Anomaly detection with multiple methods
        
        ### 🚀 **Production Ready**
        - Automated code generation for model deployment
        - Batch prediction capabilities
        - Model serialization and loading
        - Performance monitoring and evaluation
        
        ---
        
        ## 🎯 **Supported File Formats**
        - CSV files with automatic encoding detection
        - Excel files (XLSX, XLS)
        - Large datasets up to 100,000 rows
        
        ## 🔧 **Advanced Preprocessing**
        - Intelligent missing value handling
        - Outlier detection and removal
        - Automatic feature encoding (Label, One-hot)
        - Feature scaling and normalization
        - Feature selection and dimensionality reduction
        
        ---
        
        ### 📈 **Model Categories Supported**
        
        **🌳 Tree-based Models**
        - Random Forest, Extra Trees
        - Gradient Boosting, XGBoost
        - LightGBM, CatBoost
        
        **🔢 Linear Models**  
        - Logistic/Linear Regression
        - Ridge, Lasso, Elastic Net
        - Support Vector Machines
        
        **🧠 Neural Networks**
        - Multi-layer Perceptron
        - Convolutional Neural Networks (1D)
        - LSTM, GRU, Bidirectional RNNs
        
        **📊 Other Algorithms**
        - K-Nearest Neighbors
        - Naive Bayes variants
        - Discriminant Analysis
        - Gaussian Processes
        
        ---
        
        ## 🚀 **Get Started**
        Upload your dataset using the file uploader above and let our AI-powered system automatically:
        
        1. **🔍 Analyze** your data structure and quality
        2. **🧹 Clean** and preprocess your dataset intelligently  
        3. **📊 Visualize** patterns and relationships
        4. **🤖 Recommend** the best machine learning models
        5. **📦 Generate** production-ready deployment code
        
        **Ready to transform your data into insights? Upload your file now! 📂✨**
        """)
        
        # Add some example datasets info
        with st.expander("💡 Example Use Cases", expanded=False):
            st.markdown("""
            ### 🏢 **Business Applications**
            - **Sales Forecasting**: Predict future sales based on historical data
            - **Customer Churn**: Identify customers likely to leave
            - **Fraud Detection**: Detect anomalous transactions
            - **Price Optimization**: Optimize pricing strategies
            
            ### 🏥 **Healthcare & Science**  
            - **Medical Diagnosis**: Classify diseases based on symptoms
            - **Drug Discovery**: Predict molecular properties
            - **Clinical Trials**: Analyze treatment effectiveness
            
            ### 💰 **Finance**
            - **Credit Scoring**: Assess loan default risk
            - **Algorithmic Trading**: Predict stock movements
            - **Risk Management**: Quantify financial risks
            
            ### 🏭 **Manufacturing & IoT**
            - **Predictive Maintenance**: Predict equipment failures
            - **Quality Control**: Detect defective products
            - **Supply Chain**: Optimize inventory management
            """)

if __name__ == "__main__":
    main()