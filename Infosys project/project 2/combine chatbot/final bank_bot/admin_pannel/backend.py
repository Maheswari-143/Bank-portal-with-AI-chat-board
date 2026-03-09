import csv
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
app.secret_key = "bank_secret_key"

# 🔥 SAME CSV AS ADMIN PANEL
DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '..', 'admin_pannel',
                 'bank_chatbot_dataset.csv')
)

def load_dataset():
    rows = []
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_message = data.get("message", "").lower().strip()

    dataset = load_dataset()

    for row in dataset:
        if row['text'].lower().strip() == user_message:
            return jsonify({
                "reply": row["response"],
                "intent": row["intent"]
            })

    return jsonify({
        "reply": "Sorry, I can help only with banking queries.",
        "intent": "out_of_scope"
    })


if __name__ == '__main__':
    app.run(debug=True)
