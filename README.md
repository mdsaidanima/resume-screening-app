# Resume Category Prediction App

This project is a Streamlit-based application that classifies uploaded resumes into categories such as Data Science, Java Developer, Web Development, Database Administrator, and DevOps.

## Features
- Upload resumes in PDF, DOCX, or TXT format
- Extract text from the uploaded file
- Predict the most likely job category
- Show confidence scores and top-3 predictions
- Support batch resume processing
- Collect feedback to improve the model over time
- Retrain the model using feedback and starter examples

## Run locally
```bash
pip install -r requirements.txt
streamlit run notebooks/app.py
```

## Deploy
This app is ready for deployment on Streamlit Community Cloud or similar platforms that support Python apps. Make sure the repository includes the requirements file and the Streamlit app entry point.
