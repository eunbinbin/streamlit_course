import streamlit as st
import datetime
from todo import TodoDB
# pip install email-validator
from email_validator import validate_email, EmailNotValidError
import re

# DB 객체 생성, 연결
db = TodoDB()
db.connectToDatabase()

# 사이드 바
sb = st.sidebar

menu = sb.selectbox('메뉴', ['회원가입', '할일', '통계'], index=1)

if menu == '회원가입':

    ucol1, ucol2 = st.columns([6, 6])

    with ucol1:
        st.subheader('회원가입')

        with st.form(key='user_reg', clear_on_submit=True):
            user_name = st.text_input('성명', max_chars=12)
            user_gender = st.radio('성별', options=('남', '여'), horizontal=True)
            user_id = st.text_input('아이디', max_chars=12)
            col1, col2 = st.columns(2)
            user_pw = col1.text_input('비밀번호', max_chars=12, type='password')
            user_pw_chk = col2.text_input('비밀번호 확인', max_chars=12, type='password')
            user_email = st.text_input('이메일')
            user_mobile = st.text_input('휴대전화', placeholder='하이픈(-) 포함 할 것')

            submit = st.form_submit_button('가입')
            if submit:
                # 이름 한글 검증
                if re.compile('[가-힣]+').sub('', user_name):
                    st.error('성명은 한글만 입력해야 합니다.')
                    st.stop()
                # 아이디 검증
                if re.compile('[a-zA-Z0-9]+').sub('', user_id):
                    st.error('아이디는 영문자와 숫자만 입력해야 합니다.')
                    st.stop()

                # 비밀번호 확인
                if user_pw != user_pw_chk:
                    st.error('비밀번호가 일치하지 않습니다.')
                    st.stop()

                # 이메일 검증
                try:
                    user_email = validate_email(user_email).email

                except EmailNotValidError as e:
                    st.error(str(e))
                    st.stop()
                # 휴대전화 검증
                regex = re.compile('^(01)\d{1}-\d{3,4}-\d{4}$')
                phone_validation = regex.search(user_mobile.replace(' ', ''))
                if not phone_validation:
                    st.error('전화번호 형식이 올바르지 않습니다.')
                    st.stop()

                db.insertUser((
                    user_name, user_gender, user_id, user_pw, user_email,
                    user_mobile, str(datetime.datetime.now())
                ))

    with ucol2:

        st.subheader('회원목록')

        users = db.readUsers()
        for user in users:

            title = user[1]+'('+ user[3] + ')'
            with st.expander(title):
                st.write(f'{user[1]}({user[5]})')
                st.write(f'{user[2]}')
                st.write(f'{user[6]}')
                st.write(f'{user[7][:19]}')


elif menu == '할일':

    st.subheader('할일입력')

    # 할일 양식
    # 내용, 날짜, 추가 버튼
    todo_content = st.text_input('할 일', placeholder='할 일을 입력하세요.')
    col1, col2, col3 = st.columns([2, 2, 2])
    todo_date = col1.date_input('날짜')
    todo_time = col2.time_input('시간')
    completed = st.checkbox('완료')
    btn = st.button('추가')

    if btn:
        db.insertTodo((
            todo_content,
            todo_date.strftime('%Y-%m-%d'),
            todo_time.strftime('%H:%M'),
            completed,
            str(datetime.datetime.now())))
        st.experimental_rerun()

    st.subheader('할일목록')

    def change_state(*args, **kargs):
        db.updateTaskState(args)

    def change_content(*args, **kargs):
        db.updateTodoContent((args[0], st.session_state[args[1]]))

    def change_date(*args, **kargs):
        db.updateTodoDate((args[0], st.session_state[args[1]].strftime('%Y-%m-%d')))

    def change_time(*args, **kargs):
        db.updateTodoTime((args[0], st.session_state[args[1]].strftime('%H:%M')))

    def delete_todo(*args, **kargs):
        # print(type(args[0]))
        db.deleteTodo(args[0])

    todos = db.readTodos()
    for todo in todos:
        col1, col2, col3, col4, col5, col6 = st.columns([1,3,2,2,3,2])
        col1.checkbox(
            str(todo[0]),
            value=True if todo[4] else False,
            on_change=change_state,
            label_visibility='collapsed',
            args=(todo[0], False if todo[4] else True))
        col2.text_input(
            str(todo[0]),
            value=todo[1],
            on_change=change_content,
            label_visibility='collapsed',
            args=(todo[0], 'content'+str(todo[0])),
            key='content'+str(todo[0]))
        col3.date_input(
            str(todo[0]),
            value=datetime.datetime.strptime(todo[2], '%Y-%m-%d').date(),
            on_change=change_date,
            label_visibility='collapsed',
            args=(todo[0], 'date'+str(todo[0])),
            key='date'+str(todo[0]))
        col4.time_input(
            str(todo[0]),
            value=datetime.datetime.strptime(todo[3], '%H:%M').time(),
            on_change=change_time,
            label_visibility='collapsed',
            args=(todo[0], 'time'+str(todo[0])),
            key='time'+str(todo[0]))
        col5.text(todo[5][0:19])
        col6.button(
            '삭제',
            on_click=delete_todo,
            args=(todo[0], ),
            key='del' + str(todo[0])
            )