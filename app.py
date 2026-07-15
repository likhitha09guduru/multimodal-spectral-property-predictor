import os

from flask import Flask, request, render_template, jsonify

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline, PredictionInputError

application = Flask(__name__)
app = application

# Loaded lazily on first request and cached for the lifetime of this process
# (one instance per gunicorn worker), rather than rebuilt on every request.
_pipeline = None


def get_pipeline() -> PredictPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictPipeline()
    return _pipeline


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Liveness/readiness probe. Reports whether trained artifacts are
    loadable without failing the whole process if they aren't (yet)."""
    try:
        get_pipeline()._ensure_loaded()
        return jsonify(status="ok", model_loaded=True), 200
    except Exception as e:
        return jsonify(status="degraded", model_loaded=False, detail=str(e)), 503


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", results=None, error=None)

    smiles = (request.form.get("smiles") or "").strip()
    ir_spectrum_raw = request.form.get("ir_spectrum") or ""

    try:
        try:
            ir_spectrum = [float(v) for v in ir_spectrum_raw.split(",") if v.strip() != ""]
        except ValueError:
            raise PredictionInputError("IR spectrum must be comma-separated numbers.")

        if not smiles:
            raise PredictionInputError("SMILES string is required.")

        data = CustomData(smiles=smiles, ir_spectrum=ir_spectrum)
        results = get_pipeline().predict(data)

        return render_template("home.html", results=float(results[0]), error=None)

    except PredictionInputError as e:
        return render_template("home.html", results=None, error=str(e)), 400
    except FileNotFoundError as e:
        logging.error(f"Model artifacts missing: {e}")
        return render_template("home.html", results=None, error=str(e)), 503
    except CustomException as e:
        logging.error(f"Prediction failed: {e}")
        return (
            render_template(
                "home.html",
                results=None,
                error="Something went wrong while predicting. Please check your input and try again.",
            ),
            500,
        )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    # This dev server is fine for local testing; for production run behind a
    # real WSGI server, e.g.: gunicorn -w 4 -b 0.0.0.0:5000 wsgi:application
    app.run(host=host, port=port, debug=debug)
