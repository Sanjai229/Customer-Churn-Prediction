from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict_form")
def predict_form(
    request: Request,
    SeniorCitizen: int = Form(...),
    tenure: float = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
    gender: str = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    PaperlessBilling: str = Form(...)
):

    # Convert form inputs into your API format
    data = {
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        f"gender_{gender}": 1,
        f"Partner_{Partner}": 1,
        f"Dependents_{Dependents}": 1,
        f"PaperlessBilling_{PaperlessBilling}": 1
    }

    # Call same logic as API
    result = predict(data)

    # Display result inside HTML
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result
        }
    )
{% if result %}
  <h3>Prediction Result:</h3>
  <p><b>Prediction:</b> {{ result.prediction }}</p>
  <p><b>Probability:</b> {{ result.probability }}</p>
{% endif %}
