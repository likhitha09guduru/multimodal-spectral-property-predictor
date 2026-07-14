from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            contains_nitrogen=request.form.get('contains_nitrogen'),
            contains_oxygen=request.form.get('contains_oxygen'),
            contains_halogen=request.form.get('contains_halogen'),
            contains_sulfur=request.form.get('contains_sulfur'),
            ir_band_ohnh_stretch_3200_3550=float(request.form.get('ir_band_ohnh_stretch_3200_3550')),
            ir_band_ch_stretch_2850_3000=float(request.form.get('ir_band_ch_stretch_2850_3000')),
            ir_band_carbonyl_1650_1750=float(request.form.get('ir_band_carbonyl_1650_1750')),
            ir_band_aromatic_1450_1600=float(request.form.get('ir_band_aromatic_1450_1600')),
            ir_band_fingerprint_500_1500=float(request.form.get('ir_band_fingerprint_500_1500')),
            h_nmr_shift_mean=float(request.form.get('h_nmr_shift_mean')),
            h_nmr_shift_std=float(request.form.get('h_nmr_shift_std')),
            h_nmr_shift_max=float(request.form.get('h_nmr_shift_max')),
            h_nmr_peak_count=int(request.form.get('h_nmr_peak_count')),
            c_nmr_shift_mean=float(request.form.get('c_nmr_shift_mean')),
            c_nmr_shift_std=float(request.form.get('c_nmr_shift_std')),
            c_nmr_shift_max=float(request.form.get('c_nmr_shift_max')),
            c_nmr_peak_count=int(request.form.get('c_nmr_peak_count')),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")
