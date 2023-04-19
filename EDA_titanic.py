import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.font_manager as fm

mnu = st.sidebar.selectbox('선택', options=['타이타닉 분석', '타이타닉 시각화1', '타이타닉 시각화2', '기타 시각화'])

titanic = sns.load_dataset('titanic')

if mnu == '타이타닉 분석':
    st.text('타이타닉 데이터 샘플')
    st.write(titanic.head())

    st.text('타이타닉 데이터 칼럼 목록')
    st.write(titanic.columns)

    st.text('타이타닉 데이터 칼럼 상세 정보')
    buffer = io.StringIO()
    titanic.info(buf=buffer)
    st.text(buffer.getvalue())

    st.text('데이터 전체 요약 정보')
    st.dataframe(titanic.describe(include='all'), use_container_width=True)

    st.text('객실 등급(pclass)에 따른 생존율 비교')
    st.dataframe(titanic[['pclass', 'survived']].groupby(['pclass'], as_index=True).mean().sort_values(by='survived',
                                                                                                       ascending=False))

    st.text('성별(sex)에 따른 생존율 비교')
    st.dataframe(
        titanic[["sex", "survived"]].groupby(['sex'], as_index=True).mean().sort_values(by='survived', ascending=False))

    st.text('함께 승선한 형제자매와 배우자 수(sibsp)에 따른 생존율 비교')
    st.dataframe(titanic[["sibsp", "survived"]].groupby(['sibsp'], as_index=True).mean().sort_values(by='sibsp',
                                                                                                     ascending=False))

    st.text('승선한 부모와 자식 수(parch)에 따른 생존율 비교')
    st.dataframe(titanic[["parch", "survived"]].groupby(['parch'], as_index=True).mean().sort_values(by='parch',
                                                                                                     ascending=False))

elif mnu == '타이타닉 시각화1':

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.text('생존 여부(survived)에 따른 연령(age) 분포')
    g = sns.FacetGrid(titanic, col='survived')
    g.map(plt.hist, 'age', bins=20)
    st.pyplot()

    # https;//computer-science-student.tistory.com/113

    st.text('성별, 생존 여부(sex, survived)에 따른 요금(fare) 분포')
    g = sns.FacetGrid(titanic, col='sex', row='survived')
    g.map(sns.regplot, 'fare', 'age', fit_reg=False)
    g.add_legend()
    st.pyplot()

    st.text('성별, 생존 여부(sex, survived)에 따른 요금(fare), 나이(age) 분포')
    g = sns.FacetGrid(titanic, col='sex', hue='survived')
    g.map(sns.regplot, 'fare', 'age', fit_reg=False)
    g.add_legend()
    st.pyplot()

    st.text('성별, 생존 여부(sex, survived)에 따른 요금(fare), 나이(age), 등급(class) 분포')
    g = sns.FacetGrid(titanic, col='sex', row='survived', hue='survived')
    g.map(sns.regplot, 'fare', 'age', fit_reg=False)
    g.add_legend()
    st.pyplot()

    st.text('생존 여부, 등급에 따른 나이(age) 분포')
    g = sns.FacetGrid(titanic, col='survived', row='class', hue='class')
    g.map(plt.hist, 'age', bins=20)
    g.add_legend()
    st.pyplot()

    st.text('승선지(embarked)와 객실 등급(class) 따른 생존율(survived)')
    g = sns.FacetGrid(titanic, row='embarked', height=2, aspect=1.8)
    g.map(sns.pointplot, 'pclass', 'survived', 'sex', palette='deep', order=[1, 2, 3], hue_order=["male", "female"])
    g.add_legend()
    st.pyplot()

elif mnu == '타이타닉 시각화2':
    st.text('히스토그램')
    fig = plt.figure(figsize=(12, 6))

    plt.title('titanic histogram(age)')
    sns.histplot(data=titanic, x='age')
    st.pyplot(plt.gcf(), clear_figure=True)

    plt.title('titanic histogram(age-alive)')
    sns.histplot(data=titanic, x='age', hue='alive')
    st.pyplot(fig, clear_figure=True)

    plt.title('titanic histogram(age-alive-stack)')
    sns.histplot(data=titanic, x='age', hue='alive', multiple='stack')
    st.pyplot(fig, clear_figure=True)

    st.text('커널정밀도추정 함수 그래프')
    plt.title('titanic kde(age)')
    sns.kdeplot(data=titanic, x='age')
    st.pyplot(fig, clear_figure=True)

    plt.title('titanic kde(age-alive)')
    sns.kdeplot(data=titanic, x='age', hue='alive', multiple='stack')
    st.pyplot(fig, clear_figure=True)

    st.text('러그플롯')
    plt.title('titanic regplot(age)')
    sns.kdeplot(data=titanic, x='age')
    sns.rugplot(data=titanic, x='age')
    st.pyplot(fig, clear_figure=True)

    st.text('막대 그래프')
    plt.title('titanic barplot(class)')
    sns.barplot(data=titanic, x='class', y='fare')
    st.pyplot(fig, clear_figure=True)

    st.text('포인트 플롯')
    plt.title('titanic pointplot(class)')
    sns.pointplot(data=titanic, x='class', y='fare')
    st.pyplot(fig, clear_figure=True)

    st.text('박스 플롯과 바이올린 플롯의 의미를 살펴볼 것')
    st.image(
        'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb0CIEt%2FbtqCy3Aazjb%2Femj6zXRBK6UBUc8iolba81%2Fimg.png')

    st.text('박스 플롯')
    plt.title('titanic pointplot(class)')
    sns.boxplot(data=titanic, x='class', y='age')
    st.pyplot(fig, clear_figure=True)

    st.text('바이올린 플롯')
    plt.title('titanic violinplot(class-age)')
    sns.violinplot(data=titanic, x='class', y='age')
    st.pyplot(fig, clear_figure=True)

    plt.title('titanic violinplot(class-age-sex)')
    sns.violinplot(data=titanic, x='class', y='age', hue='sex', split=True)
    st.pyplot(fig, clear_figure=True)

    font_list = sorted([font.name for font in fm.fontManager.ttflist])
    fs = st.selectbox('글꼴', options=font_list)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = fs
    plt.rcParams['font.size'] = 12

    st.text('카운트 플롯')
    plt.title('titanic countplot(class)-가로')
    sns.countplot(data=titanic, x='class')
    st.pyplot(fig, clear_figure=True)

    plt.title('titanic countplot(class)-세로')
    sns.countplot(data=titanic, y='class')
    st.pyplot(fig, clear_figure=True)

    st.text('파이그래프')
    plt.title('titanic pie(class)')
    df_class = pd.DataFrame(titanic.value_counts('class').tolist(), index=titanic.value_counts('class').index.tolist(),
                            columns=['count'])
    st.dataframe(df_class)

    fig1, ax1 = plt.subplots()
    explode = [0.1, 0, 0]
    ax1.pie(titanic.value_counts('class').tolist(), labels=titanic.value_counts('class').index.tolist(),
            autopct='%.1f%%', explode=explode)
    ax1.axis('equal')
    st.pyplot(fig1)

elif mnu == '기타 시각화':

    st.text('히트맵 - 데이터관계시각화')
    flights = sns.load_dataset('flights')
    st.dataframe(flights.head())

    flights_pivot = flights.pivot(index='month', columns='year', values='passengers')
    sns.set(rc={'figure.figsize': (12, 10)})
    sns.heatmap(data=flights_pivot, cmap='hot')
    st.pyplot()

    st.text('라인플롯')
    sns.lineplot(data=flights, x='year', y='passengers')
    st.pyplot()

    st.text('산점도')
    tips = sns.load_dataset('tips')
    st.dataframe(tips.head())

    sns.scatterplot(data=tips, x='total_bill', y='tip')
    st.pyplot()

    sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time')
    st.pyplot()

    sns.regplot(data=tips, x='total_bill', y='tip')
    st.pyplot()

    sns.regplot(data=tips, x='total_bill', y='tip', ci=99)
    st.pyplot()
