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
    for idx, (model_name, model_config) in enumerate(ml_models.items()):
        status_text.text(f'Evaluating {model_name}...')
        progress_bar.progress((idx + 1) / total_models)
        
        try:
            start_time = time.time()
            
            # Get the model instance
            model = model_config['model']
            
            # 🔧 FIX: Disable parallel processing for all models
            if hasattr(model, 'n_jobs'):
                model.set_params(n_jobs=1)
            
            # For ensemble models, also set n_jobs in the model itself
            if hasattr(model, 'n_jobs'):
                model.n_jobs = 1
            
            # 🔧 FIX: Use single-threaded cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=5, scoring=scoring_metric, n_jobs=1)  # Set n_jobs=1 here too
            
            # Fit model and get test score
            model.fit(X_train, y_train)
            
            # ... rest of your existing code ...