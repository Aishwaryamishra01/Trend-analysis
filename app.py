# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------------
# LOAD DATA
# -------------------------------
st.title("📊 Viral Social Media Trend Prediction Dashboard")

df = pd.read_csv("Cleaned_Viral_Social_Media_Trends.csv")

df['Post_Date'] = pd.to_datetime(df['Post_Date'])
df = df.sort_values(by='Post_Date')

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df['Day'] = df['Post_Date'].dt.day
df['Month'] = df['Post_Date'].dt.month
df['DayOfWeek'] = df['Post_Date'].dt.dayofweek

# Encode categorical
le_platform = LabelEncoder()
le_region = LabelEncoder()
le_content = LabelEncoder()
le_hashtag = LabelEncoder()

df['Platform'] = le_platform.fit_transform(df['Platform'])
df['Region'] = le_region.fit_transform(df['Region'])
df['Content_Type'] = le_content.fit_transform(df['Content_Type'])
df['Hashtag_enc'] = le_hashtag.fit_transform(df['Hashtag'])

# -------------------------------
# STEP 1: CURRENT TRENDING
# -------------------------------
recent = df[df['Post_Date'] >= df['Post_Date'].max() - pd.Timedelta(days=30)]
trending_today = recent.groupby('Hashtag')['Views'].sum().sort_values(ascending=False)

st.subheader("🔥 Trending Today")
st.write(trending_today.head(10))

# -------------------------------
# STEP 2: CREATE TREND LABEL
# -------------------------------
df['Trend_Label'] = np.where(df['Views'].diff() > 0, 'Trending',
                     np.where(df['Views'].diff() < 0, 'Falling', 'Stable'))

df = df.dropna()

# -------------------------------
# STEP 3: TRAIN MODELS
# -------------------------------
features = ['Platform','Region','Content_Type','Day','Month','DayOfWeek','Hashtag_enc']

X = df[features]
y_views = df['Views']
y_trend = df['Trend_Label']

X_train, X_test, yv_train, yv_test = train_test_split(X, y_views, test_size=0.2, shuffle=False)
_, _, yt_train, yt_test = train_test_split(X, y_trend, test_size=0.2, shuffle=False)

# Regression model (views)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, yv_train)

# Classification model (trend)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, yt_train)

# -------------------------------
# STEP 4: FUTURE PREDICTION
# -------------------------------
future_pred = rf_reg.predict(X_test)

st.subheader("📈 Future Views Prediction (Sample)")
st.write(future_pred[:10])

# Growth logic
growth = future_pred[-1] - future_pred[0]

if growth > 0:
    st.success("Overall Future Trend: Growing 🚀")
else:
    st.error("Overall Future Trend: Falling 📉")

# -------------------------------
# STEP 5: TREND CLASSIFICATION
# -------------------------------
trend_pred = rf_clf.predict(X_test)

st.subheader("🔮 Trend Classification (Sample)")
st.write(trend_pred[:10])

# -------------------------------
# STEP 6: PATTERN UNDERSTANDING
# -------------------------------
importance = rf_reg.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

st.subheader("📊 Feature Importance")
st.write(feature_importance)

# -------------------------------
# STEP 7: USER INPUT PREDICTION
# -------------------------------
st.subheader("🧑‍💻 User Input Prediction")

platform = st.text_input("Enter Platform")
hashtag = st.text_input("Enter Hashtag")
content = st.text_input("Enter Content Type")
region = st.text_input("Enter Region")
day = st.number_input("Enter Day (1-31)", min_value=1, max_value=31, step=1)
month = st.number_input("Enter Month (1-12)", min_value=1, max_value=12, step=1)
dayofweek = st.number_input("Enter Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, step=1)

if st.button("Predict"):
    # Encode input safely
    platform_enc = le_platform.transform([platform])[0] if platform in le_platform.classes_ else 0
    region_enc = le_region.transform([region])[0] if region in le_region.classes_ else 0
    content_enc = le_content.transform([content])[0] if content in le_content.classes_ else 0
    hashtag_enc = le_hashtag.transform([hashtag])[0] if hashtag in le_hashtag.classes_ else 0

    user_data = np.array([[platform_enc, region_enc, content_enc, day, month, dayofweek, hashtag_enc]])

    pred_views = rf_reg.predict(user_data)[0]
    pred_trend = rf_clf.predict(user_data)[0]

    st.subheader("✅ Result")
    st.write("Predicted Views:", int(pred_views))
    st.write("Predicted Trend:", pred_trend)
