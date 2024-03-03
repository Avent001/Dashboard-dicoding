"""
Pertanyaan :
1. Apakah ada pola khusus pada jam-jam puncak (misalnya, jam berapa terjadi peminjaman sepeda terbanyak)?
2. Bagaimana pengaruh kelembapan dan kecepatan angin terhadap peminjaman sepeda pada jam-jam tertentu?
3. Apakah terdapat korelasi antara kondisi cuaca ekstrem dengan penurunan dalam peminjaman sepeda pada jam-jam tertentu?
4. Bagaimana prediksi permintaan peminjaman sepeda pada jam-jam tertentu ?
5. Bagaimana dampak liburan pada pola peminjaman sepeda pada jam-jam tertentu, 
dan apakah terdapat perubahan signifikan pada perilaku peminjaman pada hari libur?
"""
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px





def Every_hour(df):
    hour_counting = df.groupby("hr")["cnt"].sum()
    max_hour = hour_counting.idxmax()
    max_count = hour_counting.max()
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(hour_counting.index, hour_counting, marker="o", linestyle="-", color="b")
    ax.set_title("Counting Every Hours")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.set_xticks(range(24))
    for i, txt in enumerate(hour_counting):
        if txt == max_count:
            continue
        ax.annotate(txt, (hour_counting.index[i], txt), textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=8)
    ax.annotate(f"Time: {max_hour}\nCount: {max_count}",
                 xy=(max_hour, max_count), textcoords="offset points", xytext=(0, 10),
                 arrowprops=dict(facecolor="red", arrowstyle="->"), ha="center", fontsize=10, color="red")

    return fig, ax 

def Humidity_airspeed(df):
    fig = px.scatter_3d(df, x='hum', y='windspeed', z='cnt', color='hr', size='windspeed')
    fig.update_layout(scene=dict(
        xaxis=dict(title='Humidity'),
        yaxis=dict(title='Wind Speed'),
        zaxis=dict(title='Counting')
    ))
    return fig

def Wheater_analysis(df):
    df_analysis = pd.DataFrame(df)
    sns.set(style="whitegrid", palette="Set2")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(x="weathersit", y="cnt", data=df_analysis, ax=ax, palette="Set2", inner="quartile")
    mean_points = df_analysis.groupby("weathersit")["cnt"].mean().values
    ax.scatter(x=np.arange(len(mean_points)), y=mean_points, color="red", s=100, marker="o", label="Mean")
    ax.set_title("Analysis Bicyle in Weathers", fontsize=16)
    ax.set_xlabel("Weathers", fontsize=14)
    ax.set_ylabel("Bicyle Count", fontsize=14)
    ax.legend(title="Weathers Mean", loc="upper right")

    return fig, ax 

def Hours_prediction(df):
    X = df[["hr"]]
    y = df["cnt"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=9)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test, y_test, color='black', label='Actual Data')
    X_test_sorted, y_pred_sorted = zip(*sorted(zip(X_test.values, y_pred)))
    ax.plot(X_test_sorted, y_pred_sorted, color='blue', linewidth=3, label=f'Polynomial Regression (Degree={3})')
    ax.set_title(f'Predicted Bike Demand based on hours using Polynomial Regression')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.legend()
    return fig, ax

def Holiday_analysis(df):
    holiday_data = df[df["holiday"] == 1]
    non_holiday_data = df[df["holiday"] == 0]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(holiday_data["hr"], bins=24, kde=True, label='Holiday', color='blue')
    sns.histplot(non_holiday_data["hr"], bins=24, kde=True, label='Weekday', color='orange')
    ax.set_title(f'Distribution Holiday and Weekday')
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.legend()
    t_stat, p_value = stats.ttest_ind(holiday_data["cnt"], non_holiday_data["cnt"])
    mean_holiday = np.mean(holiday_data["cnt"])
    mean_non_holiday = np.mean(non_holiday_data["cnt"])
    std_holiday = np.std(holiday_data["cnt"])
    std_non_holiday = np.std(non_holiday_data["cnt"])
    print(f'T-statistic: {t_stat}\nP-value: {p_value}')
    print(f'Mean Holiday: {mean_holiday}, Std Holiday: {std_holiday}')
    print(f'Mean Weekday: {mean_non_holiday}, Std Weekday: {std_non_holiday}')

    return fig, ax
data = pd.read_csv("https://github.com/Avent001/Dashboard-dicoding/blob/master/Bike-sharing-dataset/hour.csv")
st.sidebar.title("Bike Sharing Analysis Dashboard")


page = st.sidebar.radio("Select a page", ["Every Hour", "Humidity & Airspeed", "Weather Analysis", "Hourly Prediction", "Holiday Analysis"])


st.title(page)

if page == "Every Hour":
    st.subheader("Bike Sharing Counting Every Hour")
    fig, _ = Every_hour(data)
    st.pyplot(fig)

elif page == "Humidity & Airspeed":
    st.subheader("Humidity and Wind Speed Analysis")
    fig = Humidity_airspeed(data)
    st.plotly_chart(fig)

elif page == "Weather Analysis":
    st.subheader("Weather Analysis")
    fig, _ = Wheater_analysis(data)
    st.pyplot(fig)

elif page == "Hourly Prediction":
    st.subheader("Hourly Bike Demand Prediction")
    fig, _ = Hours_prediction(data)
    st.pyplot(fig)

elif page == "Holiday Analysis":
    st.subheader("Holiday Analysis")
    fig, _ = Holiday_analysis(data)
    st.pyplot(fig)
