from flask import Flask, render_template, send_from_directory, request
import os
from Fetch_activity_monthly_ver import generate_prediction, plot_predictions
import calendar
import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)

figure_path = 'plot.png'
if not os.path.exists(figure_path):
    print("Generating figure...")
    plot_predictions()
    plt.savefig(figure_path)
    plt.close()

@app.route('/')
def home():
    a = "index.html"
    with open(a) as f:
        html = f.read()
    return html

@app.route('/ask_4_prediction', methods = ['POST'])
def ask_4_prediction():
    user_input = request.form['userInput']
    a = "index.html"
    with open(a) as f:
        html = f.read()
    try:
      input_date = datetime.datetime.strptime(user_input, "%Y-%m").date()
    except ValueError as e:
      html += "<h3> Invalid input. Please check yyyy-mm format.</h3>"
    else:
      earliest_date = datetime.datetime.strptime("2020-01", "%Y-%m").date()
      if input_date < earliest_date:
         html += "<h3> Invalid date. Please make sure your month is on or after 2020-01.</h3>"
      else:
        result = generate_prediction(user_input)
        html += "<h3> Estimated result: "+str(int(result[1]))+" receipts, daily "+str(int(result[0]))+" receipts</h3>"
    return html

@app.route('/plot.png')
def plot():
   return Flask.send_file('plot.png')

if __name__ == '__main__':
    app.run(debug=True)