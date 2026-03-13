# ================================================================
#   TASK 1 — SPAM EMAIL CLASSIFIER
#   Kodbud AI Internship
#   Just run: python task1_spam_classifier.py
# ================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------------
# STEP 1 : DATASET  (built-in, no download needed)
# ----------------------------------------------------------------
data = {
    'label': (['ham'] * 40 + ['spam'] * 40),
    'message': [
        # ---- HAM messages ----
        "Hey, are you free for lunch tomorrow?",
        "Can you send me the project report by tonight?",
        "Let's meet at the coffee shop at 5 pm.",
        "I'll call you back in 10 minutes.",
        "Happy birthday! Hope you have an amazing day!",
        "Did you watch the match last night?",
        "Please review the document I sent you.",
        "The meeting is rescheduled to Monday morning.",
        "What time does your flight arrive?",
        "I am running 10 minutes late, sorry!",
        "Can you pick up some groceries on your way home?",
        "Mom wants us all at dinner on Sunday.",
        "Just finished the assignment, sending it now.",
        "Are you coming to the office tomorrow?",
        "My laptop is not working, can you help?",
        "Let me know when you reach home safely.",
        "The project deadline is next Friday.",
        "I will be there in 20 minutes.",
        "Do you want to catch a movie this weekend?",
        "Please confirm your attendance for the event.",
        "Good morning! Hope you have a great day.",
        "Can we reschedule our call to 4 pm?",
        "I have forwarded the email to the manager.",
        "The internet is down at my place right now.",
        "Just checking in, how are you doing?",
        "See you at the gym at 7 am tomorrow.",
        "The package was delivered this afternoon.",
        "I forgot to bring my charger, can I borrow yours?",
        "We need to discuss the budget for next quarter.",
        "The kids are doing well in school this year.",
        "Thanks for your help yesterday, really appreciated.",
        "I will submit the form by end of day.",
        "Can you share the presentation slides please?",
        "Our internet bill is due next week.",
        "Dad called, he wants to talk to you.",
        "The cafe opens at 8 am on weekdays.",
        "I passed my driving test today!",
        "What is the address of the new office?",
        "The team lunch is at 1 pm on Thursday.",
        "Please bring your ID card to the interview.",

        # ---- SPAM messages ----
        "WINNER! You have won a 1000 pound prize! Call NOW to claim!",
        "FREE entry! Win a brand new iPhone. Text WIN to 80085.",
        "URGENT: Your bank account is suspended. Click here now.",
        "Congratulations! You are selected for a 500 dollar cash reward.",
        "You have been pre-approved for a loan. Apply immediately!",
        "Get Viagra 80 percent off! Click the link below for cheap meds.",
        "Your mobile number WON our weekly lottery! Claim today!",
        "HOT SINGLES in your area! Click now! No subscription needed!",
        "EARN 5000 dollars per week from home! Limited offer! Reply YES",
        "ALERT: Your account needs verification. Click here ASAP.",
        "You have a secret admirer! Find out who: click this link.",
        "Cheap loans approved instantly! No credit check needed!",
        "You are our LUCKY winner this month! Claim your prize.",
        "FREE gift card worth 200 dollars waiting for you. Act NOW!",
        "Your parcel could not be delivered. Pay 2 pounds to rebook.",
        "MAKE MONEY FAST! Work from home. Earn 300 dollars daily easily.",
        "Exclusive deal! Buy 1 get 3 free. Click link to redeem.",
        "Congratulations, you qualify for a FREE government grant!",
        "Double your investment in 7 days! Guaranteed returns!",
        "WARNING: Virus detected on your phone! Click to remove.",
        "You have been selected for a FREE cruise vacation. Claim!",
        "Reply STOP to unsubscribe or HELP for more free offers.",
        "Cash prize of 5000 pounds is waiting! Text CLAIM to 82277.",
        "Your credit score improved! Get your FREE report now.",
        "Lose 10 kg in 10 days! Secret diet pill doctors hate.",
        "Last chance! Your free trial expires tonight. Renew now.",
        "Someone sent you a gift. Confirm your address to receive.",
        "Your Netflix account will be closed. Update payment now.",
        "You won a FREE holiday! Just pay the small booking fee.",
        "Limited time: 90 percent off all items! Sale ends in 1 hour!",
        "Send this to 10 friends and get FREE recharge instantly.",
        "Your number was randomly selected. Claim 1000 dollars today!",
        "Cheap meds delivered to your door. No prescription needed.",
        "Earn coins just by clicking! Join our platform today FREE.",
        "Your SIM card will be blocked. Call us immediately!",
        "SHOCKING secret banks do not want you to know. Click now.",
        "Free iPhone 15 giveaway! Only 3 spots left. Sign up fast.",
        "You qualify for a home improvement grant. Apply online.",
        "IMPORTANT: Confirm your identity or your account is locked.",
        "Massive clearance sale! 95 percent off branded items. Shop now!",
    ]
}

df = pd.DataFrame(data)

# ----------------------------------------------------------------
# STEP 2 : ENCODE LABELS   (ham = 0 , spam = 1)
# ----------------------------------------------------------------
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------------------------------------------
# STEP 3 : TRAIN / TEST SPLIT
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'],
    test_size=0.2, random_state=42, stratify=df['label_num']
)

# ----------------------------------------------------------------
# STEP 4 : TF-IDF VECTORIZER
# ----------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ----------------------------------------------------------------
# STEP 5 : TRAIN NAIVE BAYES MODEL
# ----------------------------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ----------------------------------------------------------------
# STEP 6 : EVALUATE MODEL
# ----------------------------------------------------------------
y_pred = model.predict(X_test_vec)
acc    = accuracy_score(y_test, y_pred)

print("=" * 56)
print("    SPAM EMAIL CLASSIFIER  |  Kodbud AI Internship")
print("=" * 56)
print(f"\n  Dataset  : {len(df)} messages")
print(f"  Ham      : {df[df.label == 'ham'].shape[0]} messages")
print(f"  Spam     : {df[df.label == 'spam'].shape[0]} messages")
print(f"  Training : {len(X_train)} samples")
print(f"  Testing  : {len(X_test)} samples")
print(f"\n  Accuracy : {acc * 100:.1f}%\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ----------------------------------------------------------------
# STEP 7 : PREDICT NEW MESSAGES
# ----------------------------------------------------------------
print("=" * 56)
print("              LIVE PREDICTIONS")
print("=" * 56)

test_messages = [
    "Congratulations! You won a FREE iPhone! Click to claim!",
    "Hey bro, are you coming to college tomorrow?",
    "URGENT: Your bank account is suspended. Verify now!",
    "Can you send me the notes from today's lecture?",
    "You have been selected for a 1000 dollar cash prize. Reply YES",
    "Mom made biryani, come home early tonight!",
    "FREE gift card worth 500 dollars waiting for you. Act NOW!",
    "The assignment deadline is tomorrow at 11 pm.",
]

print()
for msg in test_messages:
    vec   = vectorizer.transform([msg])
    pred  = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    conf  = max(proba) * 100
    tag   = "SPAM" if pred == 1 else "HAM "
    icon  = "[!!!]" if pred == 1 else "[ ok]"
    print(f"  {icon} {tag}  ({conf:5.1f}%)  |  {msg[:48]}")

print()
print("=" * 56)
print("  Task 1 Complete! Spam Classifier working!  [OK]")
print("=" * 56)

# ----------------------------------------------------------------
# BONUS : Type your own message
# ----------------------------------------------------------------
print("\n  [ BONUS ] Type your own message below to test live!")
print("  Press Ctrl+C to exit.\n")

while True:
    try:
        user_input = input("  Your message: ").strip()
        if not user_input:
            continue
        vec   = vectorizer.transform([user_input])
        pred  = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        conf  = max(proba) * 100
        tag   = "SPAM [!!!]" if pred == 1 else "HAM  [ ok]"
        print(f"  Result  -->  {tag}  |  Confidence: {conf:.1f}%\n")
    except KeyboardInterrupt:
        print("\n\n  Goodbye! See you in Task 2!\n")
        break
