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
auth = firebase.auth()

# auth.create_user_with_email_and_password('silverbeen07@gmail.com', '111111')

isLogged = False

try:

    login = auth.sign_in_with_email_and_password('silverbeen07@gmail.com', '111111')
    user = auth.get_account_info(login['idToken'])
    #print(user['users'][0]['localId'])
    print(user)
    isLogged = True


except Exception as e:
    print(e)

if isLogged == True:
    print('로그인 성공')
else:
    print('로그인 실패')