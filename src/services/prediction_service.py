from services.classifier import Classifier
from configurations.configs import Configs
from utils.logging_util import Logger

CONFIGS = Configs.get_instance()
LOGGER = Logger.get_instance()


class PredictionService:
    """
    This class runs the prediction and returns the response.
    """

    __model = Classifier.get_instance().get_model()
    __vc = Classifier.get_instance().get_vc()

    @staticmethod
    def get_response(text: str):
        """
        This method performs the prediction and prepares the response.

        :param text: The text to be classified.
        :type text: str
        :return: The response containing the prediction result.
        :rtype: dict
        """
        # Step 01: Initialize the response.
        response = dict()
        results = dict()

        vectorized_text = dict()
        vectorized_text['test']  = (PredictionService.__vc.transform([text]))  # see options in the above cell

        print ('DONE - [EMBEDDING] Apply Chosen Embeddings to the Tweets')
        # Step 02: Predict the label/class of the received text.
        predicted_sentiment = PredictionService.__model.predict(vectorized_text['test']).tolist()

        # Step 03: Parse the prediction result.
        if (predicted_sentiment[0] == 1):
            results["label"] = "Relevant"
        else:
            results["label"] = "Not Relevant"

        # Step 04: Prepare the response.
        response["status"] = 200
        response["results"] = results

        # Step 05: Return the response.
        return response
