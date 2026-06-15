import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# 1. Load the Decision Tree model safely
MODEL_PATH = "tree_pkl.pkl"
model = None
model_features = []

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            # Try to automatically extract the features the model was trained on
            if hasattr(model, "feature_names_in_"):
                model_features = list(model.feature_names_in_)
    except Exception as e:
        print(f"Error loading model: {e}")

# Pre-defined options for the dropdowns
CATEGORICAL_OPTIONS = {
    "Gender": ["Male", "Female", "Other"],
    "Stage": ["Stage I", "Stage II", "Stage III", "Stage IV"],
    "Treatment_Type": ["Surgery", "Chemotherapy", "Radiation", "Immunotherapy", "Targeted Therapy"],
    "Cancer_Type": ["Breast Cancer", "Lung Cancer", "Colorectal Cancer", "Prostate Cancer", "Melanoma", "Leukemia"],
    "State": ["California", "New York", "Texas", "Florida", "Illinois", "Other"],
    "City": ["Los Angeles", "New York City", "Houston", "Miami", "Chicago", "Other"]
}

# 2. Fully Embedded UI Template (HTML + Tailwind CSS + JavaScript)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oncology Survival Prognosis Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; }
        .glass-panel {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
    </style>
</head>
<body class="bg-gradient-to-tr from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4 sm:p-8">

    <div class="w-full max-w-4xl glass-panel text-slate-900 rounded-3xl shadow-2xl p-6 sm:p-10 relative overflow-hidden my-6">
        <div class="absolute -top-24 -right-24 w-64 h-64 bg-indigo-400 rounded-full blur-3xl opacity-30 pointer-events-none"></div>
        <div class="absolute -bottom-24 -left-24 w-64 h-64 bg-emerald-400 rounded-full blur-3xl opacity-20 pointer-events-none"></div>

        <div class="border-b border-slate-300/60 pb-6 mb-8 text-center sm:text-left">
            <h1 class="text-3xl font-bold tracking-tight text-indigo-950">Oncology Prognosis System</h1>
            <p class="text-slate-600 mt-1 font-medium">Predictive Clinical Decision Tree Classification Engine</p>
            {% if not model_loaded %}
            <div class="mt-3 p-3 bg-rose-100 border border-rose-300 rounded-xl text-rose-800 text-sm font-semibold">
                ⚠️ System Error: 'tree_pkl.pkl' file could not be loaded. Please ensure it is uploaded alongside this script.
            </div>
            {% endif %}
        </div>

        <form id="predictionForm" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Age Profile (Years)</label>
                    <input type="number" name="Age" min="1" max="120" value="55" required
                        class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                </div>

                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Biological Gender</label>
                    <select name="Gender" class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                        {% for item in options['Gender'] %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                    </select>
                </div>

                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Cancer Site / Type</label>
                    <select name="Cancer_Type" class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                        {% for item in options['Cancer_Type'] %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                    </select>
                </div>

                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Staging Criteria</label>
                    <select name="Stage" class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                        {% for item in options['Stage'] %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                    </select>
                </div>

                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Primary Treatment Protocol</label>
                    <select name="Treatment_Type" class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                        {% for item in options['Treatment_Type'] %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                    </select>
                </div>

                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Observed Survival Timeline (Months)</label>
                    <input type="number" name="Survival_Months" min="0" max="600" value="24" required
                        class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                </div>

                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Geographic State</label>
                    <select name="State" class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                        {% for item in options['State'] %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                    </select>
                </div>

                <div class="flex flex-col">
                    <label class="text-sm font-bold text-slate-700 mb-2">Metropolitan City</label>
                    <select name="City" class="px-4 py-3 bg-white border border-slate-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium transition-all">
                        {% for item in options['City'] %}<option value="{{ item }}">{{ item }}</option>{% endfor %}
                    </select>
                </div>

            </div>

            <div class="pt-4">
                <button type="submit" class="w-full bg-indigo-950 text-white font-bold py-4 px-6 rounded-xl hover:bg-indigo-900 shadow-lg active:scale-[0.99] transition-all tracking-wide">
                    RUN PREDICTIVE ASSESSMENT
                </button>
            </div>
        </form>

        <div id="resultBox" class="hidden mt-8 p-6 rounded-2xl transition-all border">
            <div class="flex flex-col items-center text-center space-y-2">
                <span class="text-xs uppercase font-extrabold tracking-widest text-slate-500">Analysis Classification Outcome</span>
                <div id="predictionResult" class="text-4xl font-black">N/A</div>
                <p id="confidenceResult" class="text-sm font-semibold text-slate-600 mt-1"></p>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const resultBox = document.getElementById('resultBox');
            const predOut = document.getElementById('predictionResult');
            const confOut = document.getElementById('confidenceResult');
            
            const formData = new FormData(this);
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.success) {
                    resultBox.classList.remove('hidden', 'bg-rose-50', 'border-rose-200', 'bg-emerald-50', 'border-emerald-200');
                    
                    if (data.prediction.toLowerCase() === 'alive') {
                        resultBox.classList.add('bg-emerald-50', 'border-emerald-200');
                        predOut.className = "text-4xl font-black text-emerald-700";
                    } else {
                        resultBox.classList.add('bg-rose-50', 'border-rose-200');
                        predOut.className = "text-4xl font-black text-rose-700";
                    }
                    
                    predOut.innerText = data.prediction;
                    confOut.innerText = data.confidence !== "N/A" ? "Confidence metrics: " + data.confidence : "";
                } else {
                    alert("Model structural mismatch error:\\n" + data.error);
                }
            } catch (err) {
                alert("Network communication failure with the backend server.");
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, options=CATEGORICAL_OPTIONS, model_loaded=(model is not None))

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"success": False, "error": "Model file 'tree_pkl.pkl' is missing or failed to initialize."}), 500
    
    try:
        # 1. Capture incoming form parameters
        raw_input = {
            "Age": float(request.form.get("Age", 55)),
            "Gender": request.form.get("Gender"),
            "City": request.form.get("City"),
            "State": request.form.get("State"),
            "Cancer_Type": request.form.get("Cancer_Type"),
            "Stage": request.form.get("Stage"),
            "Treatment_Type": request.form.get("Treatment_Type"),
            "Survival_Months": float(request.form.get("Survival_Months", 24))
        }
        
        # 2. Build the precise DataFrame 
        df_input = pd.DataFrame([raw_input])

        # 3. ADVANCED ALIGNMENT CHECK: If the model uses One-Hot Encoding features directly
        if model_features:
            # Recreate dummy structure matching the original training frame
            df_encoded = pd.get_dummies(df_input)
            # Fill out missing structural columns with 0 values
            for col in model_features:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            # Ensure the feature column layout matches the expected tree structure exactly
            final_df = df_encoded[model_features]
        else:
            # Fallback if specific inner structural feature array properties aren't visible
            final_df = df_input

        # 4. Generate Classification Output
        raw_prediction = model.predict(final_df)[0]
        prediction_label = str(raw_prediction)
        
        # Calculate prediction metrics if model supports probability outputs
        probability_text = "N/A"
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(final_df)[0]
                classes = model.classes_
                prob_details = [f"{cls}: {prob*100:.1f}%" for cls, prob in zip(classes, probs)]
                probability_text = " | ".join(prob_details)
            except Exception:
                pass
            
        return jsonify({
            "success": True,
            "prediction": prediction_label,
            "confidence": probability_text
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"{str(e)} - Check that features match exactly."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
