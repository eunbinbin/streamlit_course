import pyrebase

firebaseconfig = {
    'apiKey': "AIzaSyDoWWqQaEbaDyLq7oe5Bsb7OVYG4Ng2_qc",
    'authDomain': "streamlit-course-gsgh.firebaseapp.com",
    'projectId': "streamlit-course-gsgh",
    'storageBucket': "streamlit-course-gsgh.appspot.com",
    'messagingSenderId': "915095814691",
    'appId': "1:915095814691:web:8cc716363c43c1810e6796",
    'databaseURL': 'https://streamlit-course-gsgh-default-rtdb.firebaseio.com/'
}

firebase = pyrebase.initialize_app((firebaseconfig))
db = firebase.database()

data1 = {'silver': {
    'name': '황은빈',
    "age":18,
    "address": '대구 북구'
}}
# print(db.push(data))
db.child("users").push(data1)

data2 = {'arirang': {
    'name': '유관순',
    "age":20,
    "address": '부산 북구'
}}
#print(db.push(data))
db.child("users").push(data2)
