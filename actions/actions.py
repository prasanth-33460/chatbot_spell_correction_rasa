from textblob import TextBlob
from rasa_sdk import Action
from rasa_sdk.events import SlotSet

class ActionCorrectSpelling(Action):
    def name(self) -> str:
        return "action_correct_spelling"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get("text")
        corrected_input = str(TextBlob(user_input).correct())
        dispatcher.utter_message(f"Corrected input: {corrected_input}")
        return [SlotSet("user_input", corrected_input)]