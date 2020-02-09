import joblib
from configurations.configs import Configs
from utils.logging_util import Logger

CONFIGS = Configs.get_instance()
LOGGER = Logger.get_instance()


class Classifier:
    """
    This class is a singleton which returns the instance of the Classifier.
    """

    __instance = None

    def __init__(self):
        """
        This method initializes the instance of the Classifier.
        """
        if Classifier.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.model = False
            self.vc = False
            self.load_model()

    def load_model(self):
        """
        This method loads the Classifier model in the instance variable 'model'.
        """
        # Step 01: Load the model path from the configurations.
        model_path = CONFIGS.get_configuration("MODEL_PATH")
        vc_path = CONFIGS.get_configuration("VC_PATH")

        # Step 02: Load the model.
        try:
            self.model = joblib.load(model_path)
            self.vc = joblib.load(vc_path)
        except Exception as exc:
            error = "Error occurred while loading the classifier model: " + str(exc)
            LOGGER.logger.exception(error)
            LOGGER.log_err.exception(error)

        # Step 03: Assign this instance to '__instance'.
        Classifier.__instance = self

        LOGGER.logger.info(
            "Finished loading the classifier model: " + str(self.get_model())
        )

    def get_model(self):
        """
        This method returns the Classifier model.

        :return: The Classifier model.
        """
        return self.model

    def get_vc(self):
        """
        This method returns the Classifier model.

        :return: The Classifier model.
        """
        return self.vc

    @staticmethod
    def get_instance():
        """
        This method instantiates an instance of the Classifier if it is not already instantiated and returns the same.

        :return: - Instance of the Classifier
        """
        if Classifier.__instance is None:
            Classifier()
        return Classifier.__instance
