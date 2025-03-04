import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS





app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "https://your-allowed-origin.com"}})


# --- Model and Utility Functions ---
def calculate_credit_score(subscriptions, travel, social_media, credit_history, age, employment, loans, transactions):
    subscription_points = (float(subscriptions) / 12) * 100
    travel_points = (float(travel) / 5) * 100
    social_media_points = (float(social_media) / 10000) * 100
    credit_history_points = (float(credit_history) - 300) / 5.5
    age_points = (float(age) - 18) / 0.62
    employment_points = 100 if employment == 'Employed' else 75 if employment == 'Self-employed' else 25
    loans_points = (5 - float(loans)) * 20
    transactions_points = (float(transactions) / 10000) * 100
    total_score = (subscription_points + travel_points + social_media_points +
                   credit_history_points + age_points + employment_points +
                   loans_points + transactions_points)
    return total_score

def generate_dataset(num_samples=1000):
    np.random.seed(42)
    subscriptions = np.random.randint(1, 13, num_samples).astype(float)
    travel = np.random.randint(0, 6, num_samples).astype(float)
    social_media = np.random.randint(100, 10001, num_samples).astype(float)
    credit_history = np.random.randint(300, 851, num_samples).astype(float)
    age = np.random.randint(18, 81, num_samples).astype(float)
    employment = np.random.choice(['Employed', 'Self-employed', 'Unemployed'], num_samples)
    loans = np.random.randint(0, 6, num_samples).astype(float)
    transactions = np.random.randint(1000, 10001, num_samples).astype(float)

    X = np.column_stack((subscriptions, travel, social_media, credit_history, age, employment, loans, transactions))
    y = np.array([calculate_credit_score(*row) for row in X])
    return X, y

# --- Prepare the dataset and train the model ---
X, y = generate_dataset()
# Encode employment: 2 for 'Employed', 1 for 'Self-employed', 0 for 'Unemployed'
X = np.column_stack((X[:, :5], np.where(X[:, 5]=='Employed', 2, np.where(X[:, 5]=='Self-employed', 1, 0)), X[:, 6:]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

feature_names = ['Subscriptions', 'Travel', 'Social Media', 'Credit History', 'Age', 'Employment', 'Loans', 'Transactions']

def predict_credit_score(subscriptions, travel, social_media, credit_history, age, employment, loans, transactions):
    employment_encoded = 2 if employment == 'Employed' else 1 if employment == 'Self-employed' else 0
    input_data = np.array([[subscriptions, travel, social_media, credit_history, age, employment_encoded, loans, transactions]])
    predicted_score = rf_model.predict(input_data)[0]

    feature_importance = rf_model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    suggestions = []
    for feature, importance in sorted_importance:
        if feature == 'Subscriptions' and subscriptions < 12:
            suggestions.append(f"Increase the number of subscriptions (current: {subscriptions})")
        elif feature == 'Travel' and travel < 5:
            suggestions.append(f"Increase travel frequency (current: {travel})")
        elif feature == 'Social Media' and social_media < 10000:
            suggestions.append(f"Increase social media interactions (current: {social_media})")
        elif feature == 'Credit History' and credit_history < 850:
            suggestions.append(f"Improve credit history (current: {credit_history})")
        elif feature == 'Employment' and employment != 'Employed':
            suggestions.append(f"Seek full-time employment (current: {employment})")
        elif feature == 'Loans' and loans > 0:
            suggestions.append(f"Reduce the number of loans (current: {loans})")
        elif feature == 'Transactions' and transactions < 10000:
            suggestions.append(f"Increase transaction volume (current: {transactions})")

    # Create a flag based on the predicted score
    flag = "Approved" if predicted_score >= 450 else "Not Approved"

    return {
        "predicted_score": predicted_score,
        "feature_importance": feature_importance_dict,
        "suggestions": suggestions,
        "flag": flag
    }


# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required_fields = ['subscriptions', 'travel', 'social_media', 'credit_history', 'age', 'employment', 'loans', 'transactions']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

    try:
        subscriptions = float(data['subscriptions'])
        travel = float(data['travel'])
        social_media = float(data['social_media'])
        credit_history = float(data['credit_history'])
        age = float(data['age'])
        employment = data['employment']
        loans = float(data['loans'])
        transactions = float(data['transactions'])
    except Exception:
        return jsonify({'error': 'Invalid data format. Please check your input.'}), 400

    result = predict_credit_score(subscriptions, travel, social_media, credit_history, age, employment, loans, transactions)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
