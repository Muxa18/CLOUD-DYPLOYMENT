# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Load model and model columns
model = joblib.load('saved_model.pkl')
model_columns = joblib.load('model_columns.pkl')  # list of column names used during training

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1) get JSON from POST
        input_json = request.get_json(force=True)
        # if client sends a list of dicts, take first row — assignment requires single-row input
        if isinstance(input_json, list):
            df = pd.DataFrame(input_json)
        elif isinstance(input_json, dict):
            df = pd.DataFrame([input_json])
        else:
            return jsonify({'error': 'Invalid JSON format. Send a JSON object with feature keys.'}), 400

        # 2) Ensure same columns / order as training
        df = df.reindex(columns=model_columns, fill_value=0)

        # 3) Predict
        preds = model.predict(df)

        # 4) Return prediction (list since `.predict` can output array)
        return jsonify({'Prediction': preds.tolist()})

    except Exception as e:
        # Useful debug info for dev; in assignment deliverable you can keep it simple
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    # debug True for development; set debug=False when handing in if you like
    app.run(host='127.0.0.1', port=5000, debug=True)
