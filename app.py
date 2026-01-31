import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string, redirect, url_for
import pickle
import pandas as pd
import io
import base64
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Load the Random Forest model and CountVectorizer from the pickle files
with open('random_forest_model.pkl', 'rb') as file:
    model_rf = pickle.load(file)
with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Sentiment Analysis</title>
            <style>
              body { background-color: #121212; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
              .container { text-align: center; max-width: 600px; padding: 20px; border-radius: 10px; background-color: #1e1e1e; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; justify-content: center; align-items: center; }
              .form-group { margin-bottom: 15px; width: 100%; display: flex; justify-content: center; }
              .form-control { width: 100%; max-width: 400px; padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #333; color: #fff; text-align: center; }
              .form-control::placeholder { color: #bbb; text-align: center; }
              .btn-primary { background-color: #007aff; border: none; padding: 10px 20px; border-radius: 5px; color: #fff; cursor: pointer; }
              .btn-primary:hover { background-color: #005bb5; }
              h1, h2 { margin-bottom: 20px; }
            </style>
          </head>
          <body>
            <div class="container">
              <h1>Sentiment Analysis</h1>
              <form action="/process" method="post" enctype="multipart/form-data">
                <div class="form-group">
                  <input type="text" class="form-control" id="review" name="review" placeholder="Enter your review">
                </div>
                <div class="form-group">
                  <input type="file" class="form-control" id="file" name="file" accept=".tsv">
                </div>
                <button type="submit" class="btn btn-primary">Submit</button>
              </form>
              {% if prediction is not none %}
              <div class="mt-3">
                <h2>Prediction: {{ prediction }}</h2>
              </div>
              {% endif %}
              {% if error_message is not none %}
              <div class="mt-3">
                <h2 style="color: red;">{{ error_message }}</h2>
              </div>
              {% endif %}
              {% if pie_chart is not none %}
              <div class="mt-3">
                <img src="data:image/png;base64,{{ pie_chart }}" alt="Pie Chart" style="max-width: 100%; height: auto;">
              </div>
              {% endif %}
            </div>
          </body>
        </html>
    ''', prediction=None, pie_chart=None, error_message=None)

@app.route('/process', methods=['POST'])
def process():
    review = request.form.get('review')
    file = request.files.get('file')

    if review:
        STOPWORDS = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        review = re.sub('[^a-zA-Z]', ' ', review).lower().split()
        review = ' '.join(stemmer.stem(word) for word in review if word not in STOPWORDS)
        X = cv.transform([review]).toarray()
        prediction = model_rf.predict(X)
        return render_template_string('''
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
                <title>Sentiment Analysis</title>
                <style>
                  body { background-color: #121212; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
                  .container { text-align: center; max-width: 600px; padding: 20px; border-radius: 10px; background-color: #1e1e1e; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; justify-content: center; align-items: center; }
                  .form-group { margin-bottom: 15px; width: 100%; display: flex; justify-content: center; }
                  .form-control { width: 100%; max-width: 400px; padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #333; color: #fff; text-align: center; }
                  .form-control::placeholder { color: #bbb; text-align: center; }
                  .btn-primary { background-color: #007aff; border: none; padding: 10px 20px; border-radius: 5px; color: #fff; cursor: pointer; }
                  .btn-primary:hover { background-color: #005bb5; }
                  h1, h2 { margin-bottom: 20px; }
                </style>
              </head>
              <body>
                <div class="container">
                  <h1>Sentiment Analysis</h1>
                  <form action="/process" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                      <input type="text" class="form-control" id="review" name="review" placeholder="Enter your review">
                    </div>
                    <div class="form-group">
                      <input type="file" class="form-control" id="file" name="file" accept=".tsv">
                    </div>
                    <button type="submit" class="btn btn-primary">Submit</button>
                  </form>
                  <div class="mt-3">
                    <h2>Prediction: {{ prediction }}</h2>
                  </div>
                </div>
              </body>
            </html>
        ''', prediction=int(prediction[0]), pie_chart=None, error_message=None)

    elif file:
        try:
            df = pd.read_csv(file, delimiter='\t')
            if 'feedback' not in df.columns:
                raise ValueError("TSV file must contain a 'feedback' column.")
        except Exception as e:
            return render_template_string('''
                <!doctype html>
                <html lang="en">
                  <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
                    <title>Sentiment Analysis</title>
                    <style>
                      body { background-color: #121212; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
                      .container { text-align: center; max-width: 600px; padding: 20px; border-radius: 10px; background-color: #1e1e1e; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; justify-content: center; align-items: center; }
                      .form-group { margin-bottom: 15px; width: 100%; display: flex; justify-content: center; }
                      .form-control { width: 100%; max-width: 400px; padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #333; color: #fff; text-align: center; }
                      .form-control::placeholder { color: #bbb; text-align: center; }
                      .btn-primary { background-color: #007aff; border: none; padding: 10px 20px; border-radius: 5px; color: #fff; cursor: pointer; }
                      .btn-primary:hover { background-color: #005bb5; }
                      h1, h2 { margin-bottom: 20px; }
                    </style>
                  </head>
                  <body>
                    <div class="container">
                      <h1>Sentiment Analysis</h1>
                      <form action="/process" method="post" enctype="multipart/form-data">
                        <div class="form-group">
                          <input type="text" class="form-control" id="review" name="review" placeholder="Enter your review">
                        </div>
                        <div class="form-group">
                          <input type="file" class="form-control" id="file" name="file" accept=".tsv">
                        </div>
                        <button type="submit" class="btn btn-primary">Submit</button>
                      </form>
                      <div class="mt-3">
                        <h2 style="color: red;">Error: {{ error_message }}</h2>
                      </div>
                    </div>
                  </body>
                </html>
            ''', prediction=None, pie_chart=None, error_message=str(e))

        positive_count = df['feedback'].sum()
        negative_count = len(df) - positive_count

        labels = 'Positive', 'Negative'
        sizes = [positive_count, negative_count]
        colors = ['#007aff', '#ff3b30']
        explode = (0.1, 0)

        fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the figure size here
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=False, startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        ax.axis('equal')
        plt.setp(ax.texts, color='white', fontweight='bold')
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#121212')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=300)
        buf.seek(0)
        pie_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return render_template_string('''
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
                <title>Sentiment Analysis</title>
                <style>
                  body { background-color: #121212; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
                  .container { text-align: center; max-width: 600px; padding: 20px; border-radius: 10px; background-color: #1e1e1e; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); display: flex; flex-direction: column; justify-content: center; align-items: center; }
                  .form-group { margin-bottom: 15px; width: 100%; display: flex; justify-content: center; }
                  .form-control { width: 100%; max-width: 400px; padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #333; color: #fff; text-align: center; }
                  .form-control::placeholder { color: #bbb; text-align: center; }
                  .btn-primary { background-color: #007aff; border: none; padding: 10px 20px; border-radius: 5px; color: #fff; cursor: pointer; }
                  .btn-primary:hover { background-color: #005bb5; }
                  h1, h2 { margin-bottom: 20px; }
                </style>
              </head>
              <body>
                <div class="container">
                  <h1>Sentiment Analysis</h1>
                  <form action="/process" method="post" enctype="multipart/form-data">
                    <div class="form-group">
                      <input type="text" class="form-control" id="review" name="review" placeholder="Enter your review">
                    </div>
                    <div class="form-group">
                      <input type="file" class="form-control" id="file" name="file" accept=".tsv">
                    </div>
                    <button type="submit" class="btn btn-primary">Submit</button>
                  </form>
                  <div class="mt-3">
                    <img src="data:image/png;base64,{{ pie_chart }}" alt="Pie Chart" style="max-width: 100%; height: auto;">
                  </div>
                </div>
              </body>
            </html>
        ''', prediction=None, pie_chart=pie_chart, error_message=None)

    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)