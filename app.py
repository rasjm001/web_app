import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import joblib
import shap
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier

app = Flask(__name__)
app.secret_key = 'malware_analysis_secret_key'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Classifier definitions
classifiers = {
    'RandomForest': {
        'model': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=4, 
                                       min_samples_leaf=2, random_state=42),
        'scale_required': False,
        'params': {
            'n_estimators': [50, 75],
            'max_depth': [3, 5, 7],
            'min_samples_split': [4, 6],
            'min_samples_leaf': [2, 3]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'scale_required': True,
        'params': {
            'n_neighbors': [7, 9, 11]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(penalty='l2', C=0.5, solver='liblinear', max_iter=1000, random_state=42),
        'scale_required': True,
        'params': {
            'C': [0.1, 0.5, 1]
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42),
        'scale_required': False,
        'params': {
            'max_depth': [3, 5],
            'min_samples_split': [6, 8],
            'min_samples_leaf': [2, 3]
        }
    },
    'SVM': {
        'model': SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42),
        'scale_required': True,
        'params': {
            'C': [0.1, 0.5, 1],
            'kernel': ['rbf']
        }
    }
}

# Attack simulation classifiers
attack_classifiers = ['LogisticRegression', 'DecisionTree', 'RandomForest']
epsilon_values = [0.0, 0.05, 0.1, 0.15, 0.2]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_category(file_name):
    if "-" in file_name:
        return file_name.split("-")[0]
    else:
        return file_name

def find_category_name(file_name):
    if "-" in file_name:
        parts = file_name.split("-")
        return parts[1] if len(parts) > 1 else file_name
    else:
        return file_name

def extract_unique_file_id(file_name):
    return file_name.rsplit('-', 1)[0]

def preprocess_data(df):
    df.fillna(method="ffill", inplace=True)
    
    df["category"] = df["Category"].apply(find_category)
    df["category_name"] = df["Category"].apply(find_category_name)
    df["unique_file_id"] = df["Category"].apply(extract_unique_file_id)
    
    meta_cols = ['Category', 'category_name', 'unique_file_id']
    df_meta = df[meta_cols].copy()
    
    le_class = LabelEncoder()
    le_category = LabelEncoder()
    le_catname = LabelEncoder()
    
    df['Class_encoded'] = le_class.fit_transform(df['Class'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['category_name_encoded'] = le_catname.fit_transform(df['category_name'])
    
    df['group_id'] = df.apply(lambda row: row['unique_file_id']
                              if row['Class'] != 'Benign'
                              else f"benign_{row.name}", axis=1)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['group_id']))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.857, random_state=42)
    val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['group_id']))
    validation_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    return train_df, validation_df, test_df, le_catname

def get_features_and_target(sub_df):
    X = sub_df.drop(columns=[
        'Category', 'Class', 'category', 'category_name',
        'Class_encoded', 'category_encoded', 'category_name_encoded',
        'unique_file_id', 'group_id'
    ])
    y = sub_df['category_name_encoded']
    return X, y

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def train_and_evaluate_model(clf_name, train_df, validation_df, test_df, le_catname):
    clf_config = classifiers[clf_name]
    clf_obj = clf_config['model']
    scale_required = clf_config['scale_required']
    param_grid = clf_config['params']
    
    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(validation_df)
    X_test, y_test = get_features_and_target(test_df)
    
    meta_cols = ['Category', 'category_name', 'unique_file_id']
    meta_test = test_df[meta_cols].copy()
    
    train_groups = train_df['group_id']
    
    if scale_required:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf_obj)
        ])
        grid = {f'clf__{param}': values for param, values in param_grid.items()}
        grid_search = GridSearchCV(
            pipeline, grid, cv=GroupKFold(n_splits=5),
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train, groups=train_groups)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        grid_search = GridSearchCV(
            clf_obj, param_grid, cv=GroupKFold(n_splits=5),
            scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train, groups=train_groups)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    
    model_path = os.path.join(app.config['RESULTS_FOLDER'], f"{clf_name}_AA_model.pkl")
    joblib.dump(best_model, model_path)
    
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    y_val_pred_labels = le_catname.inverse_transform(y_val_pred)
    y_val_labels = le_catname.inverse_transform(y_val)
    y_test_pred_labels = le_catname.inverse_transform(y_test_pred)
    y_test_labels = le_catname.inverse_transform(y_test)
    
    val_report = classification_report(y_val_labels, y_val_pred_labels, output_dict=True)
    test_report = classification_report(y_test_labels, y_test_pred_labels, output_dict=True)
    
    cm = confusion_matrix(y_test_labels, y_test_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_catname.classes_, yticklabels=le_catname.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {clf_name}')
    cm_img = fig_to_base64(plt.gcf())
    
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=42, scoring='accuracy'
    )
    
    feature_importance = {X_test.columns[i]: perm_importance.importances_mean[i] 
                          for i in range(len(X_test.columns))}
    sorted_importance = dict(sorted(feature_importance.items(), 
                                   key=lambda item: item[1], reverse=True))
    
    plt.figure(figsize=(12, 8))
    features = list(sorted_importance.keys())
    importance = list(sorted_importance.values())
    top_n = min(20, len(features))
    plt.barh(range(top_n), importance[:top_n], align='center')
    plt.yticks(range(top_n), [features[i] for i in range(top_n)])
    plt.xlabel('Permutation Importance')
    plt.title(f'Feature Importance - {clf_name}')
    plt.tight_layout()
    importance_img = fig_to_base64(plt.gcf())
    
    results = {
        'classifier_name': clf_name,
        'best_params': best_params,
        'validation_report': val_report,
        'test_report': test_report,
        'confusion_matrix': cm_img,
        'shap_images': [{"title": "Feature Importance", "img": importance_img}],
        'shap_feature_importance': sorted_importance
    }
    
    metrics_list = []
    for class_label, scores in test_report.items():
        if class_label not in ["accuracy", "macro avg", "weighted avg"]:
            metrics_list.append({
                'Class': class_label,
                'Precision': scores.get('precision', None),
                'Recall': scores.get('recall', None),
                'F1-score': scores.get('f1-score', None),
                'Support': scores.get('support', None)
            })
    
    class_metrics_df = pd.DataFrame(metrics_list)
    
    if hasattr(best_model, "predict_proba"):
        test_probs = best_model.predict_proba(X_test)
        predicted_probabilities = [round(prob[label] * 100, 2) 
                                  for prob, label in zip(test_probs, y_test_pred)]
    else:
        predicted_probabilities = [None] * len(y_test)
    
    results_test_clf = X_test.copy()
    results_test_clf['Actual_Class'] = y_test_labels
    results_test_clf['Predicted_Class'] = y_test_pred_labels
    results_test_clf['Correct'] = results_test_clf['Actual_Class'] == results_test_clf['Predicted_Class']
    results_test_clf['Prediction_Probability'] = predicted_probabilities
    results_test_clf = results_test_clf.merge(meta_test, left_index=True, right_index=True)
    
    results_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"{clf_name}_Test_Results.csv")
    results_test_clf.to_csv(results_csv_path, index=False)
    
    metrics_csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"{clf_name}_Class_Metrics.csv")
    class_metrics_df.to_csv(metrics_csv_path, index=False)
    
    return results, X_test, test_df

@app.route('/')
def index():
    trained_models = [clf for clf in attack_classifiers 
                     if os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], f"{clf}_AA_model.pkl"))]
    return render_template('index.html', classifiers=list(classifiers.keys()), 
                         attack_classifiers=attack_classifiers, epsilon_values=epsilon_values,
                         trained_models=trained_models)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'datafile' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['datafile']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        session_data = {
            'filepath': filepath,
            'filename': filename
        }
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'session.json'), 'w') as f:
            json.dump(session_data, f)
        
        flash(f'File {filename} uploaded successfully!')
        return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a CSV file.')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    selected_models = request.form.getlist('models')
    
    if not selected_models:
        flash('Please select at least one model for training.')
        return redirect(url_for('index'))
    
    try:
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
        if not os.path.exists(session_path):
            flash('No dataset found. Please upload a dataset first.')
            return redirect(url_for('index'))
        
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        filepath = session_data.get('filepath')
    except Exception as e:
        flash(f'Error loading session data: {str(e)}. Please upload a dataset again.')
        return redirect(url_for('index'))
    
    if not filepath or not os.path.exists(filepath):
        flash('Dataset file not found. Please upload again.')
        return redirect(url_for('index'))
    
    try:
        df = pd.read_csv(filepath)
        train_df, validation_df, test_df, le_catname = preprocess_data(df)
        
        joblib.dump(le_catname, os.path.join(app.config['RESULTS_FOLDER'], 'le_catname.pkl'))
        test_df.to_csv(os.path.join(app.config['RESULTS_FOLDER'], 'test_df.csv'), index=True)
        
        results = {}
        for clf_name in selected_models:
            if clf_name in classifiers:
                result, X_test, test_df = train_and_evaluate_model(clf_name, train_df, validation_df, test_df, le_catname)
                results[clf_name] = result
                X_test.to_csv(os.path.join(app.config['RESULTS_FOLDER'], f"{clf_name}_X_test.csv"), index=True)
                with open(os.path.join(app.config['RESULTS_FOLDER'], 'feature_cols.json'), 'w') as f:
                    json.dump(list(X_test.columns), f)
        
        with open(os.path.join(app.config['RESULTS_FOLDER'], 'training_results.json'), 'w') as f:
            serializable_results = {}
            for clf_name, result in results.items():
                serializable_result = result.copy()
                serializable_result['shap_feature_importance'] = {
                    k: float(v) for k, v in result['shap_feature_importance'].items()
                }
                serializable_results[clf_name] = serializable_result
            json.dump(serializable_results, f)
        
        flash(f"Trained models: {', '.join(selected_models)}")
        return redirect(url_for('results'))
    
    except Exception as e:
        flash(f'Error during training: {str(e)}')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    try:
        results_path = os.path.join(app.config['RESULTS_FOLDER'], 'training_results.json')
        if not os.path.exists(results_path):
            flash('No training results found. Please train a model first.')
            return redirect(url_for('index'))
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        if not results:
            flash('Training results are empty. Please train a model again.')
            return redirect(url_for('index'))
        
        return render_template('results.html', results=results)
    except Exception as e:
        flash(f'Error loading results: {str(e)}')
        return redirect(url_for('index'))

@app.route('/attack', methods=['POST'])
def attack_simulation():
    selected_models = request.form.getlist('attack_models')
    selected_epsilons = request.form.getlist('epsilons')
    selected_epsilons = [float(eps) for eps in selected_epsilons]
    
    if not selected_models or not selected_epsilons:
        flash('Please select at least one model and one epsilon value for attack simulation.')
        return redirect(url_for('index'))
    
    try:
        le_catname = joblib.load(os.path.join(app.config['RESULTS_FOLDER'], 'le_catname.pkl'))
        test_df = pd.read_csv(os.path.join(app.config['RESULTS_FOLDER'], 'test_df.csv'), index_col=0)
        with open(os.path.join(app.config['RESULTS_FOLDER'], 'feature_cols.json'), 'r') as f:
            feature_cols = json.load(f)
        
        trained_models = {}
        required_models = selected_models + ['LogisticRegression'] if 'LogisticRegression' not in selected_models else selected_models
        
        session_path = os.path.join(app.config['UPLOAD_FOLDER'], 'session.json')
        if not os.path.exists(session_path):
            flash('No dataset found. Please upload a dataset first.')
            return redirect(url_for('index'))
        
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        filepath = session_data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            flash('Dataset file not found. Please upload again.')
            return redirect(url_for('index'))
        
        df = pd.read_csv(filepath)
        train_df, validation_df, test_df, le_catname = preprocess_data(df)
        
        for clf in required_models:
            model_path = os.path.join(app.config['RESULTS_FOLDER'], f"{clf}_AA_model.pkl")
            if os.path.exists(model_path):
                trained_models[clf] = joblib.load(model_path)
            elif clf == 'LogisticRegression':
                flash('Training LogisticRegression model for FGSM attack...')
                result, X_test, test_df = train_and_evaluate_model('LogisticRegression', train_df, validation_df, test_df, le_catname)
                trained_models['LogisticRegression'] = joblib.load(model_path)
                X_test.to_csv(os.path.join(app.config['RESULTS_FOLDER'], f"LogisticRegression_X_test.csv"), index=True)
                with open(os.path.join(app.config['RESULTS_FOLDER'], 'feature_cols.json'), 'w') as f:
                    json.dump(list(X_test.columns), f)
            else:
                flash(f'Model {clf} not found. Please train the model first.')
                return redirect(url_for('index'))
        
        X_test = pd.read_csv(os.path.join(app.config['RESULTS_FOLDER'], f"{selected_models[0]}_X_test.csv"), index_col=0)
        
        np.random.seed(42)
        
        conti_encoded = le_catname.transform(['Conti'])[0]
        conti_indices = test_df[test_df['category_name_encoded'] == conti_encoded].index
        if len(conti_indices) < 100:
            flash(f"Only {len(conti_indices)} Conti samples available. Using all available samples.")
            selected_indices = conti_indices
        else:
            selected_indices = np.random.choice(conti_indices, 100, replace=False)
        
        X_test_conti = X_test.loc[selected_indices]
        y_test_conti = test_df.loc[selected_indices, 'category_name_encoded']
        
        original_category = test_df.loc[selected_indices, 'Category']
        original_class = test_df.loc[selected_indices, 'Class']
        original_category_type = test_df.loc[selected_indices, 'category']
        original_category_name = test_df.loc[selected_indices, 'category_name']
        
        original_samples_df = X_test_conti.copy()
        original_samples_df['Category'] = original_category.values
        original_samples_df['Class'] = original_class.values
        original_samples_df['category'] = original_category_type.values
        original_samples_df['category_name'] = original_category_name.values
        original_samples_path = os.path.join(app.config['RESULTS_FOLDER'], 'original_fgsm_samples.csv')
        original_samples_df.to_csv(original_samples_path, index=False)
        
        features_not_to_modify = [
            "malfind.uniqueInjections", "malfind.protection", "malfind.commitCharge",
            "pslist.avg_threads", "malfind.ninjections", "pslist.avg_handlers",
            "psxview.not_in_deskthrd_false_avg", "pslist.nppid", "pslist.nproc",
            "svcscan.kernel_drivers", "psxview.not_in_eprocess_pool_false_avg",
            "callbacks.ngeneric", "callbacks.nanonymous", "pslist.nprocs64bit",
            "psxview.not_in_eprocess_pool", "psxview.not_in_ethread_pool",
            "psxview.not_in_pslist", "psxview.not_in_pspcid_list", "psxview.not_in_session",
            "psxview.not_in_pspcid_list_false_avg", "callbacks.ncallbacks",
            "psxview.not_in_session_false_avg", "psxview.not_in_ethread_pool_false_avg",
            "psxview.not_in_deskthrd", "psxview.not_in_csrss_handles_false_avg",
            "psxview.not_in_pslist_false_avg", "psxview.not_in_csrss_handles",
            "pslist.avg_handlers", "psxview.not_in_deskthrd_false_avg", "pslist.nppid"
        ]
        
        missing_features = [feat for feat in features_not_to_modify if feat not in X_test.columns]
        if missing_features:
            flash(f"Warning: The following features not to modify are missing: {missing_features}")
        
        indices_not_to_modify = [X_test.columns.get_loc(feat) for feat in features_not_to_modify if feat in X_test.columns]
        
        attack_results = {}
        source_model = 'LogisticRegression'
        attack_results[source_model] = {}
        
        scaler = trained_models[source_model].named_steps.get('scaler', None)
        model_clf = trained_models[source_model].named_steps['clf'] if scaler else trained_models[source_model]
        
        X_test_conti_scaled = scaler.transform(X_test_conti) if scaler else X_test_conti.values
        
        classifier = SklearnClassifier(model=model_clf)
        benign_encoded = le_catname.transform(['Benign'])[0]
        y_target = np.full(len(X_test_conti_scaled), benign_encoded)
        
        all_labels = np.arange(len(le_catname.classes_))
        
        for eps in selected_epsilons:
            try:
                fgsm = FastGradientMethod(estimator=classifier, eps=eps, targeted=True)
                
                if eps == 0.0:
                    X_test_adv_raw = X_test_conti.values
                else:
                    X_test_adv_scaled = fgsm.generate(X_test_conti_scaled, y=y_target)
                    X_test_adv_final = X_test_adv_scaled.copy()
                    X_test_adv_final[:, indices_not_to_modify] = X_test_conti_scaled[:, indices_not_to_modify]
                    X_test_adv_raw = scaler.inverse_transform(X_test_adv_final) if scaler else X_test_adv_final
                
                X_test_adv_df = pd.DataFrame(X_test_adv_raw, columns=feature_cols)
                X_test_adv_df.iloc[:, indices_not_to_modify] = X_test_conti.iloc[:, indices_not_to_modify].values
                
                X_test_adv_df['Category'] = original_category.values
                X_test_adv_df['Class'] = original_class.values
                
                original_features = X_test_conti.iloc[:, indices_not_to_modify].values
                modified_features = X_test_adv_df.iloc[:, indices_not_to_modify].values
                difference = np.abs(original_features - modified_features)
                max_diff = np.max(difference)
                
                columns_order = ['Category'] + feature_cols + ['Class']
                X_test_adv_df = X_test_adv_df[columns_order]
                
                csv_filename = f"modified_fgsm_samples_{source_model}_eps_{eps}.csv"
                csv_path = os.path.join(app.config['RESULTS_FOLDER'], csv_filename)
                X_test_adv_df.to_csv(csv_path, index=False)
                
                # Evaluate metrics for each selected model
                X_test_adv_features_only = X_test_adv_df[feature_cols]
                misclassification_results = []
                metrics = {}
                
                for clf_name in selected_models:
                    model = trained_models[clf_name]
                    
                    # Predict on original samples (eps=0.0) or adversarial samples
                    y_pred_adv = model.predict(X_test_adv_features_only)
                    
                    # Calculate metrics
                    adv_accuracy = accuracy_score(y_test_conti, y_pred_adv)
                    if eps == 0.0:
                        clean_accuracy = adv_accuracy
                    else:
                        # Load clean accuracy from stored results or recompute
                        clean_csv = os.path.join(app.config['RESULTS_FOLDER'], 
                                               f"misclassification_distribution_{source_model}_eps_0.0.csv")
                        if os.path.exists(clean_csv):
                            clean_df = pd.read_csv(clean_csv)
                            clean_accuracy = clean_df[clean_df['Eval_Model'] == clf_name]['Clean_Accuracy'].iloc[0]
                        else:
                            y_pred_clean = model.predict(X_test_conti)
                            clean_accuracy = accuracy_score(y_test_conti, y_pred_clean)
                    accuracy_drop = clean_accuracy - adv_accuracy
                    
                    evasion_count = sum(y_pred_adv != y_test_conti)
                    benign_misclassified = sum(y_pred_adv == benign_encoded)
                    
                    # Prediction distribution
                    pred_counts = pd.Series(y_pred_adv).value_counts()
                    pred_dist = pd.DataFrame({
                        'Variant': le_catname.inverse_transform(pred_counts.index),
                        'Count': pred_counts.values
                    })
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test_conti, y_pred_adv, labels=all_labels)
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=le_catname.classes_, yticklabels=le_catname.classes_)
                    plt.title(f'Confusion Matrix for {clf_name} (Epsilon={eps})')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    cm_img = fig_to_base64(plt.gcf())
                    
                    # Store metrics
                    metrics[clf_name] = {
                        'clean_accuracy': float(clean_accuracy),
                        'adversarial_accuracy': float(adv_accuracy),
                        'accuracy_drop': float(accuracy_drop),
                        'evasion_count': int(evasion_count),
                        'benign_misclassified': int(benign_misclassified),
                        'confusion_matrix': cm_img,
                        'prediction_distribution': pred_dist.to_dict(orient='records')
                    }
                    
                    # Store detailed misclassification results
                    for _, row in pred_dist.iterrows():
                        misclassification_results.append({
                            'Source_Model': source_model,
                            'Epsilon': eps,
                            'Eval_Model': clf_name,
                            'Variant': row['Variant'],
                            'Count': int(row['Count']),
                            'Evasion_Count': evasion_count,
                            'Benign_Count': benign_misclassified if row['Variant'] == 'Benign' else 0,
                            'Clean_Accuracy': float(clean_accuracy),
                            'Adversarial_Accuracy': float(adv_accuracy),
                            'Accuracy_Drop': float(accuracy_drop)
                        })
                
                # Save misclassification distribution
                misclassification_df = pd.DataFrame(misclassification_results)
                misclassification_csv = f"misclassification_distribution_{source_model}_eps_{eps}.csv"
                misclassification_df.to_csv(os.path.join(app.config['RESULTS_FOLDER'], misclassification_csv), index=False)
                
                # Store attack results for this epsilon
                attack_results[source_model][eps] = {
                    'max_diff': float(max_diff),
                    'csv_filename': csv_filename,
                    'misclassification_csv': misclassification_csv,
                    'metrics': metrics
                }
                
            except Exception as e:
                attack_results[source_model][eps] = {
                    'error': str(e),
                    'max_diff': None,
                    'csv_filename': None,
                    'misclassification_csv': None,
                    'metrics': {}
                }
        
        # Save overall attack results
        with open(os.path.join(app.config['RESULTS_FOLDER'], 'attack_results.json'), 'w') as f:
            json.dump(attack_results, f)
        
        flash(f"Attack simulation completed using {source_model} for FGSM. Metrics computed for selected models.")
        return redirect(url_for('attack_results'))
    
    except Exception as e:
        flash(f'Error during attack simulation: {str(e)}')
        return redirect(url_for('index'))

@app.route('/attack_results')
def attack_results():
    try:
        attack_results_path = os.path.join(app.config['RESULTS_FOLDER'], 'attack_results.json')
        if not os.path.exists(attack_results_path):
            flash('No attack simulation results found. Please run an attack simulation first.')
            return redirect(url_for('index'))
        
        with open(attack_results_path, 'r') as f:
            attack_results = json.load(f)
        
        if not attack_results:
            flash('Attack simulation results are empty. Please run an attack simulation again.')
            return redirect(url_for('index'))
        
        return render_template('attack_results.html', attack_results=attack_results)
    except Exception as e:
        flash(f'Error loading attack results: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download_file/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    flash('File not found.')
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)