# 🏆 KaggleForge

**AI-Powered Kaggle Competition Assistant** — No API keys, no rate limits, runs 100% locally.

KaggleForge (formerly Datathon Solver) is your local AI companion for data science competitions. It combines the local reasoning capabilities of Large Language Models (via Ollama) with automated machine learning (via PyCaret) to help you analyze, strategize, and generate baseline submissions rapidly—all within a beautiful Streamlit UI.

## Features
- **Local AI Chat:** Discuss competition rules, strategies, and feature engineering locally. No API costs.
- **AutoML Integration:** Automatically trains and compares 15+ machine learning models using PyCaret.
- **Privacy-First:** Your data never leaves your machine. Perfect for private or sensitive datasets.
- **One-Click Baseline:** Upload your data, paste the rules, and get a `submission.csv` in minutes.

---

## 🚀 How to Run on Another Computer

If you are cloning this repository to a new computer, follow these steps to get everything up and running.

### Prerequisites
- **Python 3.10+** installed on the machine.
- **Git** installed.
- ~4GB free disk space (for the LLM model + Python dependencies).
- *Optional but recommended:* NVIDIA GPU for faster LLM inference.

### Step-by-Step Setup

**1. Clone the Repository**
Open your terminal or command prompt and clone the project:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
*(Note: Replace the URL with your actual GitHub repository URL once uploaded).*

**2. Set Up a Virtual Environment (Recommended)**
It's best practice to use a virtual environment to isolate the project's dependencies.
```bash
# Create the virtual environment
python -m venv venv

# Activate it on Windows
venv\Scripts\activate

# Activate it on macOS/Linux
source venv/bin/activate
```

**3. Install Python Dependencies**
With your virtual environment active, install the required packages:
```bash
pip install -r requirements.txt
```

**4. Install & Set Up Ollama (For Local AI)**
- Download and install Ollama from [ollama.com](https://ollama.com).
- Open a new terminal window and pull the required model (e.g., Llama 3.2):
```bash
ollama pull llama3.2:3b
```
- Ensure the Ollama server is running (it usually starts automatically after installation, or you can run `ollama serve`).

---

## 💻 Usage

1. **Start the Application**
   Make sure your virtual environment is active, then run:
   ```bash
   streamlit run app.py
   ```
2. **Access the Web UI**
   Your browser should automatically open to `http://localhost:8501`.
3. **Compete!**
   - **Upload** your competition files (`train.csv`, `test.csv`, `sample_submission.csv`).
   - **Paste** the competition rules/description.
   - **Click** "Generate Solution" to let PyCaret train 15+ models.
   - **Download** your `submission.csv`.
   - **Chat** with the local AI about how to improve your score.

---

## Architecture / Tech Stack

| Component | What It Does | Cost |
|-----------|-------------|------|
| **Ollama** | Understands rules, suggests strategy, chat | Free (local) |
| **PyCaret** | Trains & compares 15+ ML models automatically | Free (open source) |
| **Streamlit** | Beautiful interactive web UI | Free (open source) |
