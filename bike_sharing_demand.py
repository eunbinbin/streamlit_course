import io

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import calendar

data_path = 'bike_sharing_demand/'
train = pd.read_csv(data_path + 'train.csv', parse_dates=['datetime'])
test = pd.read_csv(data_path + 'test.csv', parse_dates=['datetime'])
submission = pd.read_csv(data_path + 'sampleSubmission.csv', parse_dates=['datetime'])


def feature_engineering():
    train['year'] = train['datetime'].dt.year
    train['month'] = train['datetime'].dt.month
    train['day'] = train['datetime'].dt.day
    train['hour'] = train['datetime'].dt.hour
    train['minute'] = train['datetime'].dt.minute
    train['second'] = train['datetime'].dt.second
    train['date'] = train['datetime'].dt.date
    train['dayofweek'] = train['datetime'].dt.dayofweek
    train['season'] = train['season'].map({1: 'spring',
                                           2: 'summer',
                                           3: 'fall',
                                           4: 'winter'})

    train['weather'] = train['weather'].map({1: 'clear',
                                             2: 'mist, few clouds',
                                             3: 'light snow, rain, thunderstorm',
                                             4: 'heavy rain, thunderstorm, snow, fog'})
    train['weekday'] = train['date'].apply(lambda date: calendar.day_name[datetime.combine(date, datetime.min.time()).weekday()])


st.title('Bike Sharing Demand')

mnu = st.sidebar.selectbox('메뉴', options=['설명', 'EDA', '시각화', '모델링'])

if mnu == '설명':
    st.subheader('요구사항')
    st.write('''
    자전거 공유 시스템은 회원 가입, 대여 및 자전거 반납 프로세스가 도시 전역의 키오스크 위치 네트워크를 통해 자동화되는 자전거 대여 수단입니다. 이러한 시스템을 사용하여 사람들은 한 위치에서 자전거를 빌리고 필요에 따라 다른 위치에 반납할 수 있습니다. 현재 전 세계적으로 500개가 넘는 자전거 공유 프로그램이 있습니다. 이러한 시스템에서 생성된 데이터는 여행 기간, 출발 위치, 도착 위치 및 경과 시간이 명시적으로 기록되기 때문에 연구자에게 매력적입니다. 따라서 자전거 공유 시스템은 도시의 이동성을 연구하는 데 사용할 수 있는 센서 네트워크로 기능합니다. 이 대회에서 참가자는 워싱턴 DC의 Capital Bikeshare 프로그램에서 자전거 대여 수요를 예측하기 위해 과거 사용 패턴과 날씨 데이터를 결합해야 합니다.
    ''')
    st.write('''
    2년 동안의 시간당 임대 데이터가 제공됩니다. 이 대회의 경우 훈련 세트는 매월 첫 19일로 구성되고 테스트 세트는 20일부터 월말까지입니다. 대여 기간 전에 사용할 수 있는 정보만 사용하여 테스트 세트에서 다루는 각 시간 동안 대여한 총 자전거 수를 예측해야 합니다.''')
    st.image('https://storage.googleapis.com/kaggle-competitions/kaggle/3948/media/bikes.png')
    st.markdown('#### 데이터 필드')
    st.markdown('**datetime** - 기록 일시(1시간 간격)')
    st.markdown('**season** - 계절(1: 봄, 2: 여름, 3: 가을, 4: 겨울)')
    st.markdown('**holiday** - 공휴일 여부(0: 공휴일 아님, 1: 공휴일)')
    st.markdown('**workingday** - 근무일 여부(0: 근무일 아님, 1: 근무일)')
    st.markdown('**weather** - 날씨')
    st.markdown('&emsp;1: 맑음')
    st.markdown('&emsp;2: 옅은 안개, 약간 흐림')
    st.markdown('&emsp;3: 약간의 눈, 약간의 비와 천둥번개, 흐림')
    st.markdown('&emsp;4: 폭우와 천둥번개, 눈과 짙은 안개')
    st.markdown('**temp** - 실제 온도(섭씨)')
    st.markdown('**atemp** - 체감 온도(섭씨)')
    st.markdown('**humidity** - 상대 습도')
    st.markdown('**windspeed** - 풍속')
    st.markdown('**casual** - 등록되지 않은 사용자(비회원) 수')
    st.markdown('**registered** - 등록된 사용자(회원) 수')
    st.markdown('**count** - 자전거 대여량')

elif mnu == 'EDA':

    st.subheader('EDA')

    st.markdown('- (훈련 데이터 shape, 테스트 데이터 shape)')
    st.text(f'({train.shape}), ({test.shape})')

    st.markdown('- 훈련 데이터')
    st.dataframe(train.head())

    st.markdown('-피처 엔지니어링')
    st.text('시각화에 적합한 형태로 피처 변환')
    feature_engineering()
    st.code('''
    train['year'] = train['datetime'].dt.year
    train['month'] = train['datetime'].dt.month
    train['day'] = train['datetime'].dt.day
    train['hour'] = train['datetime'].dt.hour
    train['minute'] = train['datetime'].dt.minute
    train['second'] = train['datetime'].dt.second
    train['date'] = train['datetime'].dt.date
    train['dayofweek'] = train['datetime'].dt.dayofweek
    train['season'] = train['datetime'].dt.season
    train['weather'] = train['weather'].map({1: 'clear',
    2: 'mist, few clouds',
    3: 'light snow, rain, thunderstorm',
    4: 'heavy rain, thunderstorm, snow, fog'})
    train['weekday'] = train['date'].apply(lambda date: calendar.day_name[datetime.combine(date, datetime.min.time()).weekday()])
    ''')

    st.markdown('- 수정된 훈련 데이터')
    st.dataframe(train.head())

    st.markdown('- 테스트 데이터')
    st.dataframe(test.head())

    st.markdown('- 제출 데이터')
    st.dataframe(submission.head())

    st.markdown('- train.info()')
    buffer = io.StringIO()
    train.info(buf=buffer)
    st.text(buffer.getvalue())

    buffer.truncate(0)
    st.markdown('- test.info()')
    test.info(buf=buffer)
    st.text(buffer.getvalue())

elif mnu == '시각화':

    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    st.set_option('deprecation.showPyplotGlobalUse', False)

    feature_engineering()

    st.subheader('시각화')

    st.markdown('**:blue[그래프를 해석하세요.]**')

    mpl.rc('font', size=12)
    st.markdown('- count의 분포도')
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.histplot(train['count'], ax=axes[0])
    sns.histplot(np.log(train['count']), ax=axes[1])
    fig.set_size_inches(10,5)
    st.pyplot()
    st.write('원본 count값의 분포가 왼쪽으로 많이 편향되어 있어서 로그변환을 통해 정규분포에 가깝게 만듦.')

    st.markdown('- 년, 월, 일, 시간, 분, 초에 따른 대여량 평균치')
    mpl.rc('font', size=15)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(18,13)

    sns.barplot(data=train, x="year", y="count", ax=ax1)
    sns.barplot(data=train, x="month", y="count", ax=ax2)
    sns.barplot(data=train, x="day", y="count", ax=ax3)
    sns.barplot(data=train, x="hour", y="count", ax=ax4)
    sns.barplot(data=train, x="minute", y="count", ax=ax5)
    sns.barplot(data=train, x="second", y="count", ax=ax6)

    ax1.set(title="Rental amounts by year")
    ax2.set(title="Rental amounts by month")
    ax3.set(title="Rental amounts by day")
    ax4.set(title="Rental amounts by hour")

    st.pyplot()

    st.markdown('- 시즌별, 시간별, 근무일/휴무일에 따른 대여량 평균치')
    mpl.rc('font', size=15)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(18,13)

    sns.boxplot(data=train, x='season', y='count', ax=ax1)
    sns.boxplot(data=train, x='weather', y='count', ax=ax2)
    sns.boxplot(data=train, x='holiday', y='count', ax=ax3)
    sns.boxplot(data=train, x='workingday', y='count', ax=ax4)

    ax1.set(title='BoxPlot on Count Across Season')
    ax2.set(title='BoxPlot on Count Across Weather')
    ax3.set(title='BoxPlot on Count Across Holiday')
    ax4.set(title='BoxPlot on Count Across Workingday')

    st.pyplot()

    st.markdown('- 근무일, 공휴일, 요일, 계절, 날씨에 따른 시간대별 평균 대여 수량')
    mpl.rc('font', size=8)
    fig, axes = plt.subplots(nrows=5)
    plt.tight_layout()
    fig.set_size_inches(7, 13)

    sns.pointplot(x='hour', y='count', data=train, hue='workingday', ax=axes[0])
    sns.pointplot(x='hour', y='count', data=train, hue='holiday', ax=axes[1])
    sns.pointplot(x='hour', y='count', data=train, hue='weekday', ax=axes[2])
    sns.pointplot(x='hour', y='count', data=train, hue='season', ax=axes[3])
    sns.pointplot(x='hour', y='count', data=train, hue='weather', ax=axes[4])
    st.pyplot()

    st.markdown('- 온도, 체감 온도, 풍속, 습도별 대여 수량 산점도 그래프')
    mpl.rc('font', size=12)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    plt.tight_layout()
    fig.set_size_inches(7,6)
    sns.regplot(x="temp", y="count", data=train, ax=ax1, scatter_kws={'alpha':0.2}, line_kws={'color': 'blue'})
    sns.regplot(x="atemp", y="count", data=train, ax=ax2, scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'})
    sns.regplot(x="windspeed", y="count", data=train, ax=ax3, scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'})
    sns.regplot(x="humidity", y="count", data=train, ax=ax4, scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'})
    st.pyplot()

    st.markdown('- 피처 간 상관관계 매트릭스')
    corrMat = train[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    sns.heatmap(corrMat, annot=True)
    ax.set(title='Heatmap of Numerical Data')
    st.pyplot()

    st.markdown('#### 분석 정리 및 모델링 전략')
    st.markdown('**1. 타깃값 변환:** 분포도 확인 결과 타깃값인 count가 0근처로 치우쳐 있으므로 로그변환하여 정규분포에 가깝게 만들어야한다. '
                '마지막에 다시 지수변환해 count로 복원해야 한다.')
    st.markdown('**2. 파생피처 추가:** datetime 피처는 여러 가지 정보의 혼합체이므로 '
                '각각을 분리해 year, month, day, hour, minute, second 피처를 생성할 수 있다.')
    st.markdown('**3. 파생피처 추가:** datetime에 숨어 있는 또 다른 정보인 요일(weekday) 피처를 추가한다.')
    st.markdown('**4. 피처 제거:** 테스트 데이터에 없는 피처는 훈련에 사용해도 큰 의미가 없다. '
                '따라서 훈련 데이터에만 있는 casual과 registered 피처는 제거한다.')
    st.markdown('**5. 피처 제거:** datetime 피처는 인덱스 역할만 하므로 타깃값 예측에 아무런 도움이 되지 않는다.')
    st.markdown('**6. 피처 제거:** date 피처가 제공하는 정보도 year, month, day 피처에 담겨 있다.')
    st.markdown('**7. 피처 제거:** month는 season 피처의 세부 분류로 볼 수 있다.')
    st.markdown('**8. 피처 제거:** 막대 그래프 확인 결과 day는 분별력이 없다.')
    st.markdown('**9. 피처 제거:** 막대 그래프 확인 결과 minute와 second에는 아무런 정보가 담겨 있지 않다.')
    st.markdown('**10. 이상치 제거:** 포인트 플롯 확인 결과 weather가 4인 데이터는 이상치이다.')
    st.markdown('**11. 피처 제거:** 산점도 그래프와 히트맵 확인 결과'
                'windspeed 피처에는 결측값이 많고 대여 수량과의 상관관계가 매우 약하다.')

elif mnu == '모델링':

    data_path = 'bike_sharing_demand/'
    train = pd.read_csv(data_path + 'train.csv')
    test = pd.read_csv(data_path + 'test.csv')
    submission = pd.read_csv(data_path + 'sampleSubmission.csv')

    st.markdown('#### 피처 엔지니어링')
    st.markdown('**이상치 제거**')
    st.write('폭우, 번개 속에서 실제로 자전거를 대여했을 수도 있지만 이 한 건의 데이터가 머신러닝 훈련에 부정적 영향을 끼치기 때문에 제거하는 것이 좋습니다.')
    train = train[train['weather'] != 4]
    st.code("train = train[train['weather']!=4]")

    st.markdown('**데이터 합치기**')
    st.write('훈련 데이터와 테스트 데이터에 같은 피처 엔지니어링을 적용하기 위해 두 데이터를 합친다.')
    all_data = pd.concat([train, test], ignore_index=True)
    st.code("all_data = pd.concat([train, test])")

    st.dataframe(all_data)

    st.markdown('**파생 피처 추가**')

    all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])
    all_data['year'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[0])
    all_data['month'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[1])
    all_data['hour'] = all_data['datetime'].apply(lambda x: x.split()[0].split(':')[0])
    all_data['weekday'] = all_data['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').weekday())
    st.code('''
    # 날짜 피처 생성
    all_data['data'] = all_data['datetime'].apply(lambda x: x.split([0])
    # 연도 피처 생성
    all_data['year'] = all_data['datetime'].apply(lambda x: x.split([0].split('-')[0])
    # 월 피처 생성
    all_data['month'] = all_data['datetime'].apply(lambda x: x.split([0].split('-')[1])
    # 시간 피처 생성
    all_data['hour'] = all_data['datetime'].apply(lambda x: x.split([0].split(':')[0])
    # 요일 피처 생성
    all_data['weekday'] = all_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday())
    ''')
    st.write('훈련데이터는 매달 1일부터 19일까지의 기록이고, 테스트 데이터는 매달 20일부터 월말까지의 기록이다.'
             '그러므로 대여 수량을 예측할 떄 일(day) 피처는 사용할 필요가 없다. minute와 second 피처도 '
             '모든 기록이 같은 값이므로 예측에 사용할 필요가 없다.')

    st. markdown('**필요 없는 피처 제거**')
    drop_features = ['casual', 'registered', 'datetime', 'date', 'windspeed', 'month']
    all_data = all_data.drop(drop_features, axis=1)

    st.code('''
    drop_features = ['casual', 'registered', 'datetime', 'date', 'windspeed', 'month']
    all_data = all_data.drop(drop_features, axis=1)
    ''')

    st.markdown('**피처 선택이란?**  ')
    st.markdown('모델링 시 데이터의 특징을 잘 나타내는 주요 피처만 선택하는 작업을 **피처 선택(feature selection)** 이라고 한다.')
    st.markdown('피처 선택은 머신러닝 모델 성능에 큰 영향을 준다. 타깃값 예측과 관련없는 피처가 많다면 오히려 성능이 떨어진다.')

    st.markdown('**데이터 나누기**')

    train = all_data[~pd.isnull(all_data['count'])]
    test = all_data[pd.isnull(all_data['count'])]

    X_train = train.drop(['count'], axis=1)
    X_test = test.drop(['count'], axis=1)
    Y_train = train['count']

    st.code('''
    train = all_data[~pd.isnull(all_data['count'])]
    test = all_data[pd.isnull(all_data['count'])]
    
    X_train = train.drop(['count'], axis=1)
    X_test = test.drop(['count'], axis=1)
    Y_train = train['count']
    ''')

    st.dataframe(X_train.head())

    st.markdown('#### 평가지표 계산 함수 작성')
    st.write('머신러닝 훈련이 제대로 이루어졌는지 확인하려면 대상 능력을 평가할 수단, 즉 평가지표가 필요하다.')
    st.write('요구한 평가지표인 RMSLE를 계산하는 함수부터 만든다.')
    st.markdown('$\displaystyle\sqrt{{1\over{N}}\sum_{i=1}^{N}(\log{(y_{i}+1)}-\log{{y_{i}+1)})^2}$')
    import numpy as np

    def rmsle(y_true, y_pred, convertExp=True):

        if convertExp:
            y_true = np.exp(y_true)
            y_pred = np.exp((y_pred))

        log_true = np.nan_to_num(np.log(y_true+1))
        log_pred = np.nan_to_num(np.lop(y_pred+1))

        output = np.sqrt(np.mean((log_true - log_pred)**2))
        return output

    st.code('''
    import numpy as np
    
    def rmsle(y_true, y_pred, convertExp=True):
    
        if convertExp:
            y_true = np.exp(y_true)
            y_pred = np.exp((y_pred))
            
        log_true = np.nan_to_num(np.log(y_true+1))
        log_pred = np.nan_to_num(np.log(y_pred+1))
        
        output = np.sqrt(np.mean((log_true - log_pred)**2))
        return output
    ''')

    st.markdown('#### 모델 훈련')

    st.write('사이킷런이 제공하는 가장 간단한 선형 회귀 모델인 **LinearRegression 모델** 을 사용한다.')
    st.write('선형회귀에 대한 개념을 이해하려면 아래 링크의 문서를 읽어 보기 바란다.')
    st.markdown('[위키피디아: 선형회귀](https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%ED%9A%8C%EA%B7%80)')
    st.write('선형회귀는 값을 예측하는 경우 주로 사용된다.')

    from sklearn.linear_model import LinearRegression

    linear_reg_model = LinearRegression()

    log_y = np.log(Y_train)
    linear_reg_model.fit(X_train, log_y)

    st. code('''
    from sklearn.linear_model import LinearRegression
    
    linear_reg_model = LinearRegression()

    log_y = np.log(Y_train) # 타깃값 로그 변환
    linear_reg_model.fit(X_train, log_y) # 모델 훈련
    ''')
    st.markdown('')
    st.markdown('확실하게 짚어보고 갑시다.')
    st.markdown('---')
    st.markdown('**훈련 :** 피처(독립변수)와 타깃값(종속변수)이 주어졌을 때 최적의 가중치(회귀계수)를 찾는 과정')
    st.markdown('**예측 :** 최적의 가중치를 아는 상태(훈련된 모델)에서 새로운 독립변수(데이터)가 주어졌을 때 타깃값을 추정하는 과정')

    st.markdown('')
    st.markdown('**탐색적 데이터 분석 :** 예측에 도움이 될 피처를 추리고, 적절한 모델링 방법을 탐색하는 과정')
    st.markdown('**피처 엔지니어링 :** 추려진 피처들을 훈련에 적합하도록, 성능 향상에 도움되도록 가공하는 과정')
    st.markdown('---')

    st.markdown('#### 모델 성능 검증')

    preds = linear_reg_model.predict(X_train)
    rmsle_value = rmsle(log_y, preds, True)

    st.code('''
    preds = linear_reg_model.predict(X_train)
    rmsle_value = rmsle(log_y, preds, True)
    ''')

    st.write(f'선형 회귀의 RMSLE 값: {rmsle_value:.4f}')

    st.write('')
    st.markdown('#### 예측 및 결과 제출')

    st.write('1. 테스트 데이터로 예측한 결과를 이용해야 한다.')
    st.write('2. 예측한 값에 지수변환을 해줘야 한다.')

    lineararg_preds = linear_reg_model.predict(X_test)
    submission['count'] = np.exp(lineararg_preds)
    submission.to_csv('submission.csv', index=False)

    st.code('''
    # 테스트 데이터로 예측
    lineararg_preds = linear_reg_model.predict(X_test)
    # 지수 변환
    submission['count'] = np.exp(lineararg_preds)
    # 파일로 저장
    submission.to_csv('submission.csv', index=False)
    ''')

    df_s = pd.read_csv('submission.csv')
    st.dataframe(df_s)