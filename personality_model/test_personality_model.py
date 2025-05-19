from personality_model import PersonalityClassifier

# avoinding transformer warnings
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# loading the classifier
model = PersonalityClassifier()

# example
text = """
I am leader and group-leader in scouts. This means a lot of responsibility falls upon me and when issues present I have to find solutions. 
Once, there was quite a big miscommunication issue with the money we ask parents for summer camp. 
Some co-leaders had sent an unfinished draft of the camp invitation to the parents, in which the price for camp was way too low. 
This resulted in plenty of parents not paying enough. The leaders in this situation only realized their mistake when camp was already on its way, 
and they did not have enough money to provide food for the entire camp. They contacted me with the question of what they should do. 
Of course, it is annoying to ask parents for more money when they have already sent their kids to us. 
Communication with parents had never been a strong skill of mine, but now I had no choice but to contact them all with a difficult question. 
I decided the best we could do was own up our mistake, show the parents we would try and solve it ourselves but also let them know that they could still help. 
We crafted some baskets with snacks and crafts with the kids and went to sell them on the streets close to our camp location, 
but also let the parents know they could reserve some for after camp was over. They would pay in advance, and get the basket upon return of their kids. 
This way, we did not have to straight up ask more money from the parents, but gave them something in return and also made it into something fun for the kids..
"""

# predictions
results = model.predict_all_traits(text)

print("\nPredicted Personality Traits:")
for trait, label in results.items():
    print(f"{trait}: {label}")