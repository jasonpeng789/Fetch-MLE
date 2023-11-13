from flask import Flask, render_template, request
import os
import sys
from werkzeug.utils import secure_filename

# Import prediction functions from core module
sys.path.append('core')
from core.prediction import prediction_ar, prediction_lstm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web/uploads'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            selected_model = request.form['model']

            if selected_model == 'ar':
                predictions = prediction_ar(filepath, 3)
            else:
                predictions = prediction_lstm(filepath)
            months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            month_predictions = zip(months, predictions)
            return render_template('results.html', month_predictions=month_predictions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
