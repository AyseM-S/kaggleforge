"""
Prompt Templates for Ollama AI Brain
All prompts used to interact with the local LLM.
"""

SYSTEM_PROMPT = """You are a Kaggle Grandmaster AI assistant specialized in machine learning competitions.
You help analyze competition data, understand rules, suggest strategies, and explain results.
You have deep expertise in:
- Data science and machine learning
- Feature engineering
- Model selection and hyperparameter tuning
- Competition strategy and best practices

Be concise, practical, and focused on winning the competition.
When analyzing data, pay attention to:
- Missing values and how to handle them
- Feature types (categorical, numerical, ordinal)
- Target variable distribution
- Potential data leakage
- Evaluation metric implications
"""

ANALYSIS_PROMPT = """Analyze this Kaggle competition and provide a strategy.

COMPETITION RULES / DESCRIPTION:
{rules}

DATA SUMMARY:
{data_summary}

Based on the above, provide:
1. **Problem Understanding**: What is this competition asking us to predict?
2. **Key Observations**: Important patterns in the data (missing values, class imbalance, etc.)
3. **Recommended Preprocessing**: What preprocessing steps should we apply?
4. **Evaluation Metric**: What metric should we optimize for? (based on competition rules)
5. **Strategy**: Brief recommended approach.

Format your response clearly with the numbered sections above. Be concise and practical.
"""

EXPLAIN_PROMPT = """Here are the results from an AutoML model comparison on a Kaggle competition:

COMPETITION CONTEXT:
{rules}

DATA SUMMARY:
{data_summary}

MODEL RESULTS:
{model_results}

Best Model: {best_model}

Please explain:
1. Why the best model likely performed well on this data
2. Key insights from the model comparison
3. Suggestions for potential improvement (feature engineering, ensembling, etc.)

Be concise and practical.
"""

CHAT_PROMPT = """You are helping with a Kaggle competition. Here's the context:

COMPETITION: {rules}

DATA SUMMARY:
{data_summary}

CURRENT SOLUTION STATUS:
{solution_status}

The user has a question. Answer it helpfully and concisely, focusing on practical competition advice.
"""
