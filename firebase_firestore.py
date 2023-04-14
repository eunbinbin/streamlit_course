import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Setup
cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)

db=firestore.client()
docs = db.collection('users').where('name', '==', '황은빈').get()
for doc in docs:
    print(doc.to_dict())