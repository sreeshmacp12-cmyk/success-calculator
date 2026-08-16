from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('eurovision_model.pkl')
model_columns = joblib.load('model_columns.pkl')

voting_df = pd.read_excel('dataset.xlsx')


@app.route('/')
def home():
    countries = sorted(voting_df['Country'].dropna().unique())
    return render_template('index.html', countries=countries)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Song.In.English': request.form.get('Song.In.English'),
            'Group.Solo': request.form.get('Group.Solo'),
            'Artist.gender': request.form.get('Artist.gender'),
            'danceability': float(request.form.get('danceability')),
            'energy': float(request.form.get('energy'))
        }
        
        input_df = pd.DataFrame([user_input])
        
        # One-Hot Encoding
        input_encoded = pd.get_dummies(input_df)
        
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
        prediction = model.predict(input_encoded)[0]
        prediction_result = round(prediction, 2)
        
        countries = sorted(voting_df['Country'].dropna().unique())
        return render_template('index.html', prediction=prediction_result, countries=countries)

@app.route('/voting_history', methods=['POST'])
def voting_history():
    if request.method == 'POST':
        selected_country = request.form.get('country')
        
        filtered_df = voting_df[voting_df['Country'] == selected_country]
        top_giver = filtered_df.groupby('Country')['Points'].sum().idxmax()
        
        countries = sorted(voting_df['Country'].dropna().unique())
        return render_template('index.html', 
                               best_friend=top_giver, 
                               selected_country=selected_country, 
                               countries=countries)


if __name__ == '__main__':
    app.run(debug=True)