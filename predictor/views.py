
import json
import joblib
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "churn_model.joblib"

model = joblib.load(MODEL_PATH)


@csrf_exempt
@require_POST
def predict(request):
    try:
        data = json.loads(request.body)

        required_fields = [
            "tenure_months",
            "monthly_charges",
            "support_calls",
        ]

        for field in required_fields:
            if field not in data:
                return JsonResponse(
                    {"error": f"Missing field: {field}"},
                    status=400,
                )

        values = [
            float(data["tenure_months"]),
            float(data["monthly_charges"]),
            float(data["support_calls"]),
        ]

        prediction = int(model.predict([values])[0])

        return JsonResponse({
            "prediction": prediction,
            "result": (
                "Likely to churn"
                if prediction == 1
                else "Not likely to churn"
            ),
        })

    except (ValueError, TypeError, json.JSONDecodeError):
        return JsonResponse(
            {"error": "Please provide valid numeric values."},
            status=400,
        )