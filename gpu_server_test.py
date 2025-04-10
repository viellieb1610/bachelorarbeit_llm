from ollama import chat, ChatResponse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

def classify_review(review_text):
    prompt = initial_prompt + f'\n\nInput:"{review_text}"\nOutput:'
    response = chat(model=model, messages=[{"role": "user", "content": prompt}])
    response = response["message"]["content"].strip()
    if response == "Reference to other reviews":
        return 1
    elif response == "No reference":
        return 0
    else:
        return 2

df = pd.read_csv("yelp_labeled_2k.csv", nrows=100)
review = df[['review_text','has_references']]

initial_prompt = """
Your task is to classify online consumer reviews from the food rating platform Yelp. If the review mentions or refers to other reviews, classify it as “Reference to other reviews”. Otherwise classify it as “No reference”. Notice that references to other sources than Yelp do not count as references.

Examples:
Input: “Tried take out since this is only 5 star sushi nearby on yelp so I was interested. Definitely good sushi and fresh so no complaints and had to join in with the other 5 star raters. Have some unique roll options that help it stand out from some other sushi places nearby.”
Output: Reference to other reviews

Input: “So we found this place through Yelp!  Amazing!! Stand up only. Super charming. Our meal is listed below In order of what we liked best. I got photos of all but one of the dishes. If you like informal food adventures, go here!! 1. Earl Grey Creme Brûlée 2. Chicken skewers on toast with peppers and leeks - we had seconds 3. Calamari with broccoli 3. Octopus - cold with celery and red peppers, capers and red onion. No photo 4. Fried shrimp with wasabi tartar sauce 5. Sea scallop carpaccio with saffron sauce”
Output: Reference to other reviews

Input: “Not sure if I understand the hype. Meatballs were ok Sauce was bland Bread was good Mozz was bland The staff outnumbers the tables Service was good but it was more of a hangout for the employees I guess the concept worked because they seem to be doing great I'm just not a fan..”
Output: Reference to other reviews

Input: “Food was deee-lish here!   Came here to celebrate a friends birthday last night (Friday). We had such a great time! We made reservations for 10:00pm (you will not be seated until your entire party arrives) we were seated right away. This place is hip. Greatttt DJ had a mix of everything (latin music bachata,merengue,reggaeton,techno,hip hop,pop). Had the appetizers family style and yes i have to agree this plate consists of everything being fried but good! lol had the soup sancocho it was ok. the entree. I had the churrasco (skirt steak) with mash potatoes OMG deeee lish! we ordered a pitcher of red sangria due to the high praise for it on here, but i was sadly dissapointed on that part! : (  wasn't sweet at all! a bit strong.The desserts were great coconut flan, rum raisin bread pudding, guava cheesecake,and a chocolate coffee mousse cake. The decor was so cute orange cushioned leather on one half of wall and hot pink on the other side and rustic chandeliers all over. There was lounge below we didn't go there but doesnt seem like a spot to dance in only to chill without the food. They also offer doggy bags for your left overs! yeaaaah! Overall great place! :)”
Output: Reference to other reviews

Input: ”I don't get full from eating at Munchies but they are just for snacks. The service is pretty good. There was one day that they forgot 2 of the orders when I was eating there but that's ok. I was somewhat surprised that tips were already charged within the bill already.”
Output: No reference

Input: “Great Decor and layout of the restaurant. The place is well kept. As competitors to Nargis and referred by a few clients who said I MUST try it, it was good food! With improvements needed in making sure the flavors are different and not copied, it holds to it own standards. However I must say the first time I came I was denied as there was a party scheduled &amp; they refused to serve walk-ins that day. This second time I went I was able to go in and sit down. Due to the catering options not going on during my 2nd visit. The place was nearly empty. The food came out promptly, the service was standard with limited English language. Not enough welcoming vibes. Lead to my final conclusion, if nargis is booked or when they can&#39;t serve parties, come here for little different and unique food  with a more well decorated place, for bigger events!”
Output: No reference

Input: “Great bar if you enjoy low lighting and standard atmosphere. I also enjoy that they delete any and all negative reviews. Well done.”
Output: No reference
"""

model = 'llama3.1'

tqdm.pandas()
review["prediction"] = review["review_text"].progress_apply(classify_review)

print(classification_report(review['has_references'], review['prediction']))