from django.test import TestCase


class PredictionAPITest(TestCase):

    def test_prediction_with_valid_input(self):
        payload = {
            "tenure_months": 12,
            "monthly_charges": 80,
            "support_calls": 5,
        }

        response = self.client.post(
            "/predict/",
            data=payload,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json())