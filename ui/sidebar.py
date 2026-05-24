"""
Sidebar UI Component
Handles file uploads, competition description, and settings.
"""

import streamlit as st
from core.ai_brain import AIBrain


def render_sidebar():
    """Render the sidebar with uploads, settings, and Ollama status."""

    with st.sidebar:
        # ── App Title ──
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <h1 style="
                    font-size: 1.8rem;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">🏆 KaggleForge</h1>
                <p style="color: #888; font-size: 0.85rem; margin: 0.25rem 0 0 0;">
                    AI-Powered Competition Assistant
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Ollama Status ──
        st.markdown("##### 🤖 AI Status")
        connected, status_msg = AIBrain.check_connection()
        if connected:
            st.success("Ollama Connected", icon="✅")
            # Model selector
            models = AIBrain.get_available_models()
            if models:
                selected_model = st.selectbox(
                    "Select Model",
                    models,
                    index=0,
                    help="Choose which local AI model to use",
                )
                st.session_state["selected_model"] = selected_model
            else:
                st.warning("No models installed. Run: `ollama pull llama3.2:3b`")
                st.session_state["selected_model"] = None
        else:
            st.error("Ollama not running", icon="❌")
            st.caption(status_msg)
            st.code("ollama serve", language="bash")
            st.session_state["selected_model"] = None

        st.divider()

        # ── Competition Description ──
        st.markdown("##### 📋 Competition Info")
        competition_rules = st.text_area(
            "Rules & Description",
            placeholder="Paste the competition description, rules, evaluation metric, etc. here...",
            height=150,
            help="Copy the competition overview from Kaggle and paste it here",
        )
        st.session_state["competition_rules"] = competition_rules

        st.divider()

        # ── File Uploads ──
        st.markdown("##### 📁 Upload Data Files")

        train_file = st.file_uploader(
            "Training Data",
            type=["csv"],
            help="Upload train.csv",
            key="train_upload",
        )

        test_file = st.file_uploader(
            "Test Data",
            type=["csv"],
            help="Upload test.csv or test_x.csv",
            key="test_upload",
        )

        sample_sub_file = st.file_uploader(
            "Sample Submission",
            type=["csv"],
            help="Upload sample_submission.csv",
            key="sample_sub_upload",
        )

        extra_files = st.file_uploader(
            "Extra Files (optional)",
            type=["csv", "txt", "json"],
            accept_multiple_files=True,
            help="Upload any additional files",
            key="extra_upload",
        )

        # Store in session state
        st.session_state["train_file"] = train_file
        st.session_state["test_file"] = test_file
        st.session_state["sample_sub_file"] = sample_sub_file
        st.session_state["extra_files"] = extra_files

        st.divider()

        # ── Settings ──
        st.markdown("##### ⚙️ Settings")

        # Quality Mode selector (replaces dead time_budget slider)
        quality_mode = st.radio(
            "Quality Mode",
            options=["quick", "standard", "competition"],
            index=1,
            format_func=lambda x: {
                "quick": "⚡ Quick (~2 min) — Compare models only",
                "standard": "🎯 Standard (~5 min) — Compare + tune best",
                "competition": "🏆 Competition (~15 min) — Compare + tune top 3 + ensemble",
            }[x],
            help=(
                "**Quick**: Compares all algorithms, picks the best. Fast but not optimal.\n\n"
                "**Standard**: Compares + tunes the best model's hyperparameters. Good balance.\n\n"
                "**Competition**: Compares + tunes top 3 + creates a blended/stacked ensemble. "
                "Best possible score but takes longer."
            ),
        )
        st.session_state["quality_mode"] = quality_mode

        n_folds = st.slider(
            "Cross-Validation Folds",
            min_value=2,
            max_value=10,
            value=5,
            help="More folds = more reliable score estimate but slower training",
        )
        st.session_state["n_folds"] = n_folds

        metric_options = {
            "Auto (let PyCaret decide)": None,
            "Accuracy (↑ higher=better)": "Accuracy",
            "AUC (↑ higher=better)": "AUC",
            "F1 Score (↑ higher=better)": "F1",
            "R² (↑ higher=better)": "R2",
            "RMSE (↓ lower=better)": "RMSE",
            "MAE (↓ lower=better)": "MAE",
            "RMSLE (↓ lower=better)": "RMSLE",
        }
        selected_metric_label = st.selectbox(
            "Optimization Metric",
            list(metric_options.keys()),
            index=0,
            help=(
                "Choose the metric to optimize for. "
                "**↑ = higher is better** (Accuracy, AUC, F1, R²). "
                "**↓ = lower is better** (RMSE, MAE, RMSLE). "
                "Check your competition's evaluation page for which metric they use."
            ),
        )
        st.session_state["metric"] = metric_options[selected_metric_label]

        # Score Guide
        st.divider()
        st.markdown("##### 📖 Score Guide")
        st.markdown(
            """
            <div style="
                background: rgba(102, 126, 234, 0.1);
                border: 1px solid rgba(102, 126, 234, 0.2);
                border-radius: 10px;
                padding: 0.8rem;
                font-size: 0.82rem;
            ">
                <b>Understanding scores:</b><br>
                📈 <b>Accuracy, AUC, F1, R²</b>: Higher = better (1.0 = perfect)<br>
                📉 <b>RMSE, MAE, MSE</b>: Lower = better (0 = perfect)<br><br>
                💡 <b>Tip</b>: Use <b>Competition mode</b> for best scores.
                XGBoost, LightGBM, CatBoost + ensembling typically win competitions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── File status indicator ──
        st.divider()
        st.markdown("##### 📊 File Status")

        def _file_status(label, file_obj):
            if file_obj:
                st.markdown(f"✅ **{label}**: `{file_obj.name}`")
            else:
                st.markdown(f"⬜ **{label}**: not uploaded")

        _file_status("Train", train_file)
        _file_status("Test", test_file)
        _file_status("Submission", sample_sub_file)
        if extra_files:
            for ef in extra_files:
                _file_status("Extra", ef)
