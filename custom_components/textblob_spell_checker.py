from typing import Dict, Text, Any, List
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from textblob import TextBlob


# Register the component with Rasa's framework.
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=True
)
class CustomNLUComponent(GraphComponent):
    """
    Custom NLU component for performing spelling correction using TextBlob.
    This component is registered as an intent classifier in Rasa.
    """

    def __init__(self, component_config: Dict[Text, Any]) -> None:
        """
        Initialize the custom component with the given configuration.
        """
        super().__init__(component_config)
        # You can add any additional configuration processing here if necessary.
        self.threshold = component_config.get("threshold", 0.5)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """
        Create and return an instance of the component.
        """
        return cls(config)

    def train(self, training_data: TrainingData) -> Resource:
        """
        If this component requires training (e.g., using annotated training data),
        implement the training logic here. For spelling correction, no training is needed.
        """
        # No training required for this component since it's not learning any models
        return self.create_resource()

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """
        This function can be used if our component needs to augment the training data
        with additional features or tokens. Since we are doing spelling correction, no data
        augmentation is needed.
        """
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        """
        This method will be called during inference. Here, we will apply spelling correction
        on the text in the messages.
        """
        for message in messages:
            text = message.text
            # Apply TextBlob spelling correction
            blob = TextBlob(text)
            corrected_text = str(blob.correct())  # Correct spelling using TextBlob
            message.text = corrected_text  # Update the message text with corrected version

        return messages