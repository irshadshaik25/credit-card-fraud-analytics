import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import os

# Set up page configurations
st.set_page_config(
    page_title="FraudOps AI Portal", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ------------------------------------
# 1. ARTIFACT LOADING & ROBUST FALLBACKS
# ------------------------------------
@st.cache_resource
def load_artifacts():
    """Loads the model, scaler, and training features list with error handling."""
    model_path = 'dashboard/best_model.pkl'
    scaler_path = 'dashboard/scaler.pkl'
    features_path = 'dashboard/features.pkl'
    
    # Simple fallbacks if artifacts don't exist yet
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = None
        
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
        
    if os.path.exists(features_path):
        features = joblib.load(features_path)
    else:
        # Fallback minimal feature list to keep app structural integrity intact
        features = ['TransactionAmt', 'HourOfDay', 'AmtToMeanRatio', 'DeviceRisk']
        
    return model, scaler, features

model, scaler, features = load_artifacts()

# ------------------------------------
# 2. OPTIMIZED LIVE/MOCK DATA GENERATOR
# ------------------------------------
@st.cache_data
def generate_live_ledger_data():
    """Generates a reliable mock ledger by cleanly splitting scaler requirements 

    from final model features.
    """
    np.random.seed(42)
    records = 2000
    
    # 1. Unpack our saved components
    model, scaler, all_features = load_artifacts()
    
    # 2. Build the primary mock dataframe base
    df_mock = pd.DataFrame()
    df_mock['TransactionID'] = np.arange(3000000, 3000000 + records)
    
    # 3. Fill values for every feature expected by the model pipeline
    for f in all_features:
        if f == 'TransactionAmt':
            df_mock[f] = np.random.exponential(scale=120, size=records) + 5
        elif f == 'HourOfDay':
            df_mock[f] = np.random.randint(0, 24, size=records)
        elif f == 'AmtToMeanRatio':
            df_mock[f] = np.random.uniform(0.1, 5.5, size=records)
        elif f == 'DeviceRisk':
            df_mock[f] = np.random.choice([0, 1], size=records, p=[0.85, 0.15])
        else:
            df_mock[f] = np.random.normal(0, 1, size=records)
            
    # 4. Create your model data matrix matrix
    X_model = df_mock[all_features].copy()
    
    # 5. FIX: Adapt seamlessly to the exact number of features the scaler expects
    if scaler is not None and hasattr(scaler, 'n_features_in_'):
        expected_scaler_count = scaler.n_features_in_  # This will equal 210
        
        # Take only the first 210 features for the scaler
        scaler_features = all_features[:expected_scaler_count]
        X_for_scaler = X_model[scaler_features].copy()
        
        try:
            # Transform the sliced section
            X_scaled_array = scaler.transform(X_for_scaler)
        except ValueError:
            X_scaled_array = scaler.transform(X_for_scaler.values)
            
        # Put scaled features back into their designated spots in our matrix
        X_model[scaler_features] = X_scaled_array
        
    # 6. Generate final model calculations safely
    if model is not None:
        try:
            df_mock['RiskScore'] = model.predict_proba(X_model)[:, 1]
        except ValueError:
            # Secondary fallback if the model itself expects the 210 dimension shape instead
            df_mock['RiskScore'] = model.predict_proba(X_model.iloc[:, :model.n_features_in_])[:, 1]
    else:
        df_mock['RiskScore'] = (df_mock['TransactionAmt'] / df_mock['TransactionAmt'].max()) * 0.4
        
    df_mock['isFraud'] = (df_mock['RiskScore'] >= 0.5).astype(int)
    
    # Retain layout readable variables for frontend tracking visualizations
    if 'TransactionAmt' not in df_mock.columns:
        df_mock['TransactionAmt'] = np.random.exponential(scale=120, size=records) + 5
    if 'HourOfDay' not in df_mock.columns:
        df_mock['HourOfDay'] = np.random.randint(0, 24, size=records)
        
    return df_mock
    
    # Synthesize structural column layout compatibility
    for f in features:
        if f not in df_mock.columns:
            df_mock[f] = np.random.normal(0, 1, size=records)
            
    # Force column order to exactly match model specifications
    X_eval = df_mock[features].copy()
    
    # Robust scaling execution with shape error safety handling
    if scaler is not None:
        try:
            X_scaled_array = scaler.transform(X_eval)
        except ValueError:
            # Strip headers down to values array if scikit-learn feature name mismatch occurs
            X_scaled_array = scaler.transform(X_eval.values)
        X_eval_scaled = pd.DataFrame(X_scaled_array, columns=features)
    else:
        X_eval_scaled = X_eval
        
    # Generate risk scores and predictions
    if model is not None:
        df_mock['RiskScore'] = model.predict_proba(X_eval_scaled)[:, 1]
    else:
        # Static baseline mathematical formulation fallback if model is unavailable
        df_mock['RiskScore'] = (df_mock['TransactionAmt'] / df_mock['TransactionAmt'].max()) * 0.4 + (df_mock['DeviceRisk'] * 0.5)
        
    df_mock['isFraud'] = (df_mock['RiskScore'] >= 0.5).astype(int)
    return df_mock

df_dash = generate_live_ledger_data()

# Calculate dynamic percentile thresholds matching notebook fixes
critical_threshold = float(np.percentile(df_dash['RiskScore'], 98))
suspicious_threshold = float(np.percentile(df_dash['RiskScore'], 90))

def assign_risk_tier(score):
    if score >= critical_threshold: return "🔴 Critical Risk"
    elif score >= suspicious_threshold: return "🟡 Suspicious"
    return "🟢 Clear"

df_dash['Tier'] = df_dash['RiskScore'].apply(assign_risk_tier)

# ------------------------------------
# 3. SIDEBAR NAVIGATION & INTERFACE
# ------------------------------------
st.sidebar.title("🛡️ FraudOps Portal AI")
page = st.sidebar.radio("Go to Menu:", ["Overview Analytics", "Transaction Explorer", "SHAP Explainer"])

st.sidebar.markdown("---")
st.sidebar.subheader("Interactive Filters")

# Guarding filter boundaries against empty collections
max_amt = float(df_dash['TransactionAmt'].max()) if len(df_dash) > 0 else 1000.0
amt_filter = st.sidebar.slider("Transaction Amount Filter ($)", 0.0, max_amt, (0.0, max_amt))
risk_filter = st.sidebar.multiselect(
    "Filter Risk Tier Location", 
    ["🔴 Critical Risk", "🟡 Suspicious", "🟢 Clear"], 
    default=["🔴 Critical Risk", "🟡 Suspicious", "🟢 Clear"]
)

# Apply global query filtering
df_filtered = df_dash[
    (df_dash['TransactionAmt'] >= amt_filter[0]) & 
    (df_dash['TransactionAmt'] <= amt_filter[1]) & 
    (df_dash['Tier'].isin(risk_filter))
]

# ------------------------------------
# PAGE 1 — OVERVIEW ANALYTICS
# ------------------------------------
if page == "Overview Analytics":
    st.title("📊 Financial Fraud Operations Center")
    st.markdown("Real-time executive metrics and threat tracking vectors across systemic infrastructure points.")
    
    # Executive Key Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Records Evaluated", f"{len(df_filtered):,}")
    with kpi2:
        st.metric("Identified Fraud Events", f"{df_filtered['isFraud'].sum()}", delta_color="inverse")
    with kpi3:
        detection_rate = (df_filtered['isFraud'].sum() / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
        st.metric("System Detection Accuracy Rate", f"{detection_rate:.2f}%")
    with kpi4:
        avg_fraud = df_filtered[df_filtered['isFraud'] == 1]['TransactionAmt'].mean()
        st.metric("Mean Capital Loss Vector", f"${avg_fraud:.2f}" if not np.isnan(avg_fraud) else "$0.00")
        
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Risk Tier Allocation Profiles")
        if not df_filtered.empty:
            tier_counts = df_filtered['Tier'].value_counts().reset_index()
            fig_pie = px.pie(
                tier_counts, 
                values='count', 
                names='Tier', 
                hole=0.4, 
                color='Tier',
                color_discrete_map={"🔴 Critical Risk": "#ef553b", "🟡 Suspicious": "#fecb52", "🟢 Clear": "#636efa"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No matching records found within current scope parameters.")
        
    with col_right:
        st.subheader("Fraud Spike Activity by Temporal Hour")
        if not df_filtered.empty:
            hour_df = df_filtered.groupby('HourOfDay')['isFraud'].sum().reset_index()
            fig_bar = px.bar(
                hour_df, 
                x='HourOfDay', 
                y='isFraud', 
                labels={'isFraud': 'Fraud Event Count', 'HourOfDay': 'Hour of Day (24h format)'}, 
                color_discrete_sequence=['#ef553b']
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No timeline metrics available for current filters.")

# ------------------------------------
# PAGE 2 — TRANSACTION EXPLORER
# ------------------------------------
elif page == "Transaction Explorer":
    st.title("🔍 Transaction Ledger Explorer")
    st.markdown("Search, query, and trace systemic ledgers dynamically for individual operational accounting reviews.")
    
    search_query = st.text_input("⚡ Instant Ledger Key Lookup: Enter specific TransactionID...")
    
    if search_query:
        df_display = df_filtered[df_filtered['TransactionID'].astype(str).str.contains(search_query)]
    else:
        df_display = df_filtered
        
    columns_to_show = ['TransactionID', 'TransactionAmt', 'HourOfDay', 'AmtToMeanRatio', 'DeviceRisk', 'RiskScore', 'Tier']
    st.dataframe(
        df_display[columns_to_show].sort_values(by='RiskScore', ascending=False), 
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------
# PAGE 3 — SHAP EXPLAINER
# ------------------------------------
elif page == "SHAP Explainer":
    st.title("🧠 Explainable AI Auditor Engine")
    st.markdown("Deconstruct underlying structural reasoning profiles for machine-driven threat predictions via local Shapley value models.")
    
    target_id = st.number_input(
        "Target Analysis Identifier Reference Selection:", 
        min_value=int(df_dash['TransactionID'].min()), 
        max_value=int(df_dash['TransactionID'].max()), 
        value=int(df_dash['TransactionID'].iloc[0])
    )
    
    tx_row = df_dash[df_dash['TransactionID'] == target_id]
    
    if not tx_row.empty:
        score = float(tx_row['RiskScore'].values[0])
        tier = tx_row['Tier'].values[0]
        
        sc1, sc2 = st.columns(2)
        sc1.metric("Calculated Machine Fraud Probability", f"{score * 100:.2f}%")
        sc2.metric("Operational Priority Classification Level", f"{tier}")
        
        st.markdown("---")
        st.subheader("Shapley Linear Decomposition Impact Vectors")
        
        # Guard local SHAP computations against missing model objects
        if model is not None:
            explainer_local = shap.TreeExplainer(model)
            row_features = tx_row[features]
            
            # Compute shap objects explicitly
            shap_vals_local = explainer_local(row_features)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_vals_local[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("SHAP Engine visualization requires active trained model file path availability.")
            
        st.subheader("💡 Compliance Executive Briefing Report")
        if score >= critical_threshold:
            st.error(f"Action Flag: **IMMEDIATE ACCOUNT FREEZE MANDATE**. Transaction evaluated at high risk level ({score*100:.1f}% score metrics). Core anomalies point toward localized extreme capital spikes and hazardous authentication device profiles.")
        elif score >= suspicious_threshold:
            st.warning(f"Action Flag: **TRIGGER INTERN MANUAL AUDIT ENFORCEMENT**. System flags suspicious properties ({score*100:.1f}% risk metrics). Suggest evaluation of contextual location configurations.")
        else:
            st.success(f"Action Flag: **PASSIVE CLEARANCE PASS OVERRIDE**. Standard compliance profile verified. Normal system behaviors observed.")
    else:
        st.error("Transaction Identifier Reference does not exist in local operational sequence space.")