"""
Solution View UI Component
Displays data analysis, model comparison, results, and notebook download.
"""

import streamlit as st
import pandas as pd
from core.data_analyzer import DataAnalyzer
from core.ai_brain import AIBrain
from core.automl_engine import AutoMLEngine
from core.feature_engineer import FeatureEngineer
from core.notebook_generator import generate_notebook


def render_solution_view():
    """Render the main solution generation and display view."""

    # ── Header ──
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
        ">
            <h2 style="
                color: #fff;
                margin: 0 0 0.5rem 0;
                font-size: 1.6rem;
            ">🚀 Solution Generator</h2>
            <p style="
                color: rgba(255,255,255,0.6);
                margin: 0;
                font-size: 0.95rem;
            ">Upload your data, paste the rules, and let AI + AutoML do the work.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Generate Button ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        can_generate = (
            st.session_state.get("train_file") is not None
            and st.session_state.get("selected_model") is not None
        )

        quality_mode = st.session_state.get("quality_mode", "standard")
        mode_labels = {
            "quick": "⚡ Quick Solution",
            "standard": "🎯 Standard Solution",
            "competition": "🏆 Competition Solution",
        }
        button_label = mode_labels.get(quality_mode, "⚡ Generate Solution")

        generate_clicked = st.button(
            button_label,
            type="primary",
            use_container_width=True,
            disabled=not can_generate,
        )

        if not can_generate:
            if st.session_state.get("train_file") is None:
                st.caption("⬆️ Upload at least a training file to begin")
            elif st.session_state.get("selected_model") is None:
                st.caption("🤖 Start Ollama and install a model first")

    # ── Run Pipeline ──
    if generate_clicked:
        _run_full_pipeline()

    # ── Display Previous Results ──
    if st.session_state.get("pipeline_results"):
        _display_results(st.session_state["pipeline_results"])


def _run_full_pipeline():
    """Execute the full analysis + training pipeline."""
    results = {}

    # Progress container
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0, text="Starting...")

        # ── Step 1: Analyze Data ──
        progress_bar.progress(5, text="📊 Analyzing data files...")
        analyzer = DataAnalyzer()

        train_file = st.session_state["train_file"]
        test_file = st.session_state.get("test_file")
        sample_sub = st.session_state.get("sample_sub_file")
        extra_files = st.session_state.get("extra_files", [])

        analyzer.set_train(train_file.getvalue(), train_file.name)

        if test_file:
            analyzer.set_test(test_file.getvalue(), test_file.name)

        if sample_sub:
            analyzer.set_sample_submission(sample_sub.getvalue(), sample_sub.name)

        for ef in extra_files:
            if ef.name.endswith(".csv"):
                analyzer.add_extra_file(ef.getvalue(), ef.name)

        data_summary = analyzer.get_summary_text()
        full_summary = analyzer.get_full_summary()
        results["data_summary"] = data_summary
        results["full_summary"] = full_summary
        results["analyzer"] = analyzer

        progress_bar.progress(10, text="📊 Data analysis complete!")

        # ── Step 2: Feature Engineering ──
        progress_bar.progress(12, text="🔧 Engineering features...")

        target_col = full_summary.get("target_column")
        id_col = full_summary.get("id_column")
        problem_type = full_summary.get("problem_type", "classification")

        if target_col is None:
            st.error("❌ Could not detect target column. Please check your data files.")
            return

        fe = FeatureEngineer()
        train_fe, test_fe = fe.fit_transform(
            train_df=analyzer.train_df,
            target_column=target_col,
            test_df=analyzer.test_df,
            id_column=id_col,
        )
        results["fe_logs"] = fe.get_summary()
        results["train_fe"] = train_fe
        results["test_fe"] = test_fe

        progress_bar.progress(18, text="🔧 Feature engineering complete!")

        # ── Step 3: AI Analysis ──
        rules = st.session_state.get("competition_rules", "")
        model_name = st.session_state.get("selected_model", "llama3.2:3b")

        if rules.strip():
            progress_bar.progress(20, text="🧠 AI is analyzing the competition...")
            try:
                brain = AIBrain(model=model_name)
                ai_analysis = brain.analyze_competition(rules, data_summary)
                results["ai_analysis"] = ai_analysis
                brain.set_context(rules=rules, data_summary=data_summary)
                st.session_state["ai_brain"] = brain
            except Exception as e:
                results["ai_analysis"] = f"⚠️ AI analysis skipped: {e}"
                st.session_state["ai_brain"] = None
        else:
            results["ai_analysis"] = "ℹ️ No competition rules provided. Skipping AI analysis."
            try:
                brain = AIBrain(model=model_name)
                brain.set_context(data_summary=data_summary)
                st.session_state["ai_brain"] = brain
            except Exception:
                st.session_state["ai_brain"] = None

        progress_bar.progress(28, text="🧠 AI analysis complete!")

        # ── Step 4: AutoML Training ──
        quality_mode = st.session_state.get("quality_mode", "standard")
        n_folds = st.session_state.get("n_folds", 5)
        metric = st.session_state.get("metric")

        mode_labels = {
            "quick": "⚡ Quick: comparing models...",
            "standard": "🎯 Training: compare + tune best model...",
            "competition": "🏆 Competition: compare + tune top 3 + ensemble...",
        }
        progress_bar.progress(30, text=mode_labels.get(quality_mode, "Training..."))

        engine = AutoMLEngine()

        automl_results = engine.run_pipeline(
            train_df=train_fe,
            test_df=test_fe,
            target_column=target_col,
            id_column=id_col,
            problem_type=problem_type,
            metric=metric,
            quality_mode=quality_mode,
            n_folds=n_folds,
        )

        results["automl"] = automl_results
        results["engine"] = engine
        results["quality_mode"] = quality_mode

        progress_bar.progress(88, text="📝 Generating results...")

        # ── Step 5: AI Explanation ──
        if automl_results["status"] == "success" and st.session_state.get("ai_brain"):
            try:
                progress_bar.progress(90, text="🧠 AI is explaining the results...")
                explanation = st.session_state["ai_brain"].explain_results(
                    rules=rules,
                    data_summary=data_summary,
                    model_results=engine.get_comparison_text(),
                    best_model=automl_results["best_model_name"],
                )
                results["explanation"] = explanation
            except Exception as e:
                results["explanation"] = f"⚠️ AI explanation skipped: {e}"
        else:
            results["explanation"] = None

        # ── Step 6: Generate Notebook ──
        if automl_results["status"] == "success":
            progress_bar.progress(95, text="📓 Generating Jupyter notebook...")
            try:
                notebook_path = generate_notebook(
                    competition_rules=rules,
                    data_summary=data_summary,
                    target_column=target_col,
                    id_column=id_col,
                    problem_type=problem_type,
                    metric=metric,
                    quality_mode=quality_mode,
                    n_folds=n_folds,
                    best_model_name=automl_results.get("best_model_name", "N/A"),
                    best_score=automl_results.get("best_score"),
                    comparison_text=engine.get_comparison_text(),
                    feature_engineering_log=results.get("fe_logs", ""),
                    ensemble_method=automl_results.get("ensemble_method", "single"),
                    train_filename=train_file.name,
                    test_filename=test_file.name if test_file else None,
                )
                results["notebook_path"] = notebook_path
            except Exception as e:
                results["notebook_path"] = None

        progress_bar.progress(100, text="✅ Done!")

    # Store results
    st.session_state["pipeline_results"] = results
    st.session_state["data_summary_text"] = data_summary

    st.rerun()


def _display_results(results: dict):
    """Display the pipeline results."""

    # ── Feature Engineering Summary ──
    fe_logs = results.get("fe_logs", "")
    if fe_logs:
        with st.expander("🔧 Feature Engineering", expanded=False):
            st.code(fe_logs)

    # ── Data Summary ──
    with st.expander("📊 Data Profile", expanded=False):
        st.code(results.get("data_summary", "No data summary available."))

    # ── AI Analysis ──
    ai_analysis = results.get("ai_analysis")
    if ai_analysis:
        with st.expander("🧠 AI Competition Analysis", expanded=True):
            st.markdown(ai_analysis)

    # ── AutoML Results ──
    automl = results.get("automl", {})

    if automl.get("status") == "success":
        st.markdown("---")

        # Score guide banner
        problem_type = results.get("full_summary", {}).get("problem_type", "classification")
        metric_info = AutoMLEngine.get_metric_info(
            st.session_state.get("metric"), problem_type
        )

        st.markdown("### 🏆 Model Comparison Results")

        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Best Model",
                value=automl.get("best_model_name", "N/A"),
            )
        with col2:
            score = automl.get("best_score", "N/A")
            st.metric(
                label=f"Best Score {metric_info['arrow']}",
                value=score,
            )
        with col3:
            st.metric(
                label="Problem Type",
                value=problem_type.title(),
            )
        with col4:
            ensemble = automl.get("ensemble_method", "single")
            st.metric(
                label="Strategy",
                value=ensemble.title(),
            )

        # Score interpretation
        st.markdown(
            f"""
            <div style="
                background: rgba(102, 126, 234, 0.08);
                border-left: 3px solid #667eea;
                border-radius: 0 8px 8px 0;
                padding: 0.6rem 1rem;
                margin-bottom: 1rem;
                font-size: 0.85rem;
            ">
                📊 <b>Score Guide:</b> {metric_info['direction']} — {metric_info['ideal']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Comparison table
        if "comparison_df" in automl:
            st.dataframe(
                automl["comparison_df"],
                use_container_width=True,
                hide_index=True,
            )

        # Feature importance
        engine = results.get("engine")
        if engine:
            fi = engine.get_feature_importance()
            if fi is not None and len(fi) > 0:
                with st.expander("📈 Feature Importance", expanded=False):
                    st.bar_chart(
                        fi.set_index("Feature").head(15),
                        horizontal=True,
                    )

        # AI Explanation
        explanation = results.get("explanation")
        if explanation:
            with st.expander("🧠 AI Explanation of Results", expanded=True):
                st.markdown(explanation)

        # ── Downloads ──
        st.markdown("---")
        st.markdown("### 📥 Downloads")

        dl_col1, dl_col2 = st.columns(2)

        # Download submission CSV
        with dl_col1:
            if automl.get("submission_path"):
                try:
                    submission_df = pd.read_csv(automl["submission_path"])
                    st.markdown("**📄 Submission CSV**")
                    st.dataframe(submission_df.head(10), use_container_width=True)
                    with open(automl["submission_path"], "rb") as f:
                        st.download_button(
                            label="⬇️ Download submission.csv",
                            data=f.read(),
                            file_name="submission.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Error reading submission: {e}")

        # Download notebook
        with dl_col2:
            notebook_path = results.get("notebook_path")
            if notebook_path:
                try:
                    st.markdown("**📓 Solution Notebook**")
                    st.info(
                        "Complete Jupyter notebook with all code, "
                        "ready to submit to Kaggle."
                    )
                    with open(notebook_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download solution_notebook.ipynb",
                            data=f.read(),
                            file_name="solution_notebook.ipynb",
                            mime="application/x-ipynb+json",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Error reading notebook: {e}")

        # Training logs
        with st.expander("📋 Training Logs", expanded=False):
            for log in automl.get("logs", []):
                st.text(log)

    elif automl.get("status") == "error":
        st.error(f"❌ Pipeline failed: {automl.get('error', 'Unknown error')}")
        with st.expander("📋 Logs"):
            for log in automl.get("logs", []):
                st.text(log)
