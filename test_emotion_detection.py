import unittest
from unittest.mock import Mock, patch
from emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    @patch("emotion_detection.requests.post")
    def test_happy_text(self, mock_post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.01,
                    "disgust": 0.02,
                    "fear": 0.03,
                    "joy": 0.90,
                    "sadness": 0.04
                }
            }]
        }
        mock_post.return_value = response
        result = emotion_detector("I am happy")
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(result["joy"], 0.90)

    @patch("emotion_detection.requests.post")
    def test_bad_request(self, mock_post):
        response = Mock()
        response.status_code = 400
        mock_post.return_value = response
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
