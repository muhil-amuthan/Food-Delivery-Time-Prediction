# Food Delivery Time Prediction

This project contains a simple machine learning workflow for predicting food delivery times from a CSV dataset and a Flask web app for interactive predictions.

## Files

- Food_Delivery_Times.csv: sample training data
- Food_Delivery_Prediction.py: training and prediction logic
- app.py: Streamlit web interface
- delivery_model.pkl: trained model artifact
- encoders.pkl: saved label encoders
- requirements.txt: Python dependencies
- Project_Report.pdf: project report placeholder
- screenshots/: screenshots directory for app outputs

## Run locally

1. Install dependencies:
   pip install -r requirements.txt
2. Run the prediction script:
   python Food_Delivery_Prediction.py
3. Start the Flask app:
   python app.py
4. Open http://127.0.0.1:5000/
