# ============================================================
# TITANIC SURVIVAL ANALYTICS & PREDICTION SYSTEM
# ============================================================

import os
import glob
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
/* ===================== BRIGHT UI / READABILITY FIX ===================== */
:root {
    --navy: #17324d;
    --blue: #1677ff;
    --text: #17212b;
    --muted: #5d6b7a;
    --border: #d8e3ee;
    --card: #ffffff;
    --page: #f5f9fd;
}

/* Force a bright application surface even if Streamlit is using a dark theme. */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background: var(--page) !important;
    color: var(--text) !important;
}

[data-testid="stHeader"] {
    background: rgba(245,249,253,0.92) !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Global text readability */
.stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stCaptionContainer"] {
    color: var(--text) !important;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin: 12px 0 4px;
    color: #145ea8 !important;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: var(--muted) !important;
    margin-bottom: 30px;
}

.page-title {
    font-size: 34px;
    font-weight: 800;
    color: var(--navy) !important;
    margin-bottom: 5px;
}

.page-subtitle {
    font-size: 16px;
    color: var(--muted) !important;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #145ea8 !important;
    margin-top: 20px;
    margin-bottom: 12px;
}

/* All custom HTML cards are white with dark readable text. */
.info-card, .hero-card, .result-card, .login-card {
    color: var(--text) !important;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 8px 24px rgba(30, 70, 110, 0.08);
}

.info-card h1, .info-card h2, .info-card h3, .info-card h4,
.hero-card h1, .hero-card h2, .hero-card h3, .hero-card h4,
.login-card h1, .login-card h2, .login-card h3, .login-card h4,
.result-card h1, .result-card h2, .result-card h3, .result-card h4,
.info-card p, .hero-card p, .login-card p, .result-card p,
.info-card strong, .hero-card strong, .login-card strong, .result-card strong {
    color: var(--text) !important;
}

.info-card {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 18px;
}

.hero-card {
    padding: 30px 34px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.result-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #145ea8 !important;
}

.result-text {
    font-size: 17px;
    color: var(--muted) !important;
}

.login-card {
    padding: 32px;
    border-radius: 22px;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #718096 !important;
    font-size: 13px;
    margin-top: 50px;
    padding: 18px;
    border-top: 1px solid var(--border);
}

/* Bright sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input,
[data-baseweb="select"] > div, textarea {
    background: #ffffff !important;
    color: #17212b !important;
    border-color: #cbd8e6 !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder {
    color: #8291a2 !important;
    opacity: 1 !important;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 44px !important;
    border: 1px solid #b9d5f3 !important;
    background: #1677ff !important;
    color: #ffffff !important;
}
.stButton > button:hover {
    background: #0f64db !important;
    color: #ffffff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff !important;
    padding: 18px !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 6px 18px rgba(30, 70, 110, 0.06);
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: var(--text) !important;
}

/* Streamlit alert boxes: prevent white-on-white text. */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: #ffffff !important;
}
div[data-testid="stAlert"] * {
    color: #263746 !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #ffffff !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
details[data-testid="stExpander"] summary,
details[data-testid="stExpander"] p {
    color: var(--text) !important;
}

/* Form control labels */
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
[data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label {
    color: var(--text) !important;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join("models", "titanic_model.pkl")
FEATURE_PATH = os.path.join("models", "feature_columns.pkl")

try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
except Exception:
    st.error("Unable to load the trained model files.")
    st.code("models/titanic_model.pkl\nmodels/feature_columns.pkl")
    st.stop()

# ============================================================
# LOAD TITANIC DATASET
# ============================================================

@st.cache_data
def load_titanic_dataset():
    possible_files = [
        "data/titanic.csv",
        "data/Titanic.csv",
        "data/train.csv",
        "data/Titanic-Dataset.csv",
        "data/titanic_dataset.csv",
        "dataset/titanic.csv",
        "dataset/Titanic.csv",
        "Titanic.csv",
        "titanic.csv",
        "train.csv"
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                dataframe = pd.read_csv(file_path)
                if len(dataframe) > 0:
                    return dataframe, file_path
            except Exception:
                pass

    for pattern in ["data/*.csv", "dataset/*.csv", "*.csv"]:
        for file_path in glob.glob(pattern):
            try:
                dataframe = pd.read_csv(file_path)
                columns = {
                    str(column).strip().lower()
                    for column in dataframe.columns
                }
                if (
                    "survived" in columns
                    and "pclass" in columns
                    and "sex" in columns
                ):
                    return dataframe, file_path
            except Exception:
                pass

    return None, None

titanic_data, dataset_path = load_titanic_dataset()

# ============================================================
# DATASET HELPERS
# ============================================================

def find_column(dataframe, possible_names):
    if dataframe is None:
        return None

    lower_map = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    return None


def get_value(row, possible_names, default="Not Available"):
    column = find_column(titanic_data, possible_names)

    if column is None:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value

# ============================================================
# CREATE MODEL INPUT FROM REAL PASSENGER
# ============================================================

def create_model_input_from_row(row):
    pclass_column = find_column(titanic_data, ["Pclass"])
    age_column = find_column(titanic_data, ["Age"])
    sibsp_column = find_column(titanic_data, ["SibSp"])
    parch_column = find_column(titanic_data, ["Parch"])
    fare_column = find_column(titanic_data, ["Fare"])
    sex_column = find_column(titanic_data, ["Sex"])
    embarked_column = find_column(titanic_data, ["Embarked"])

    def numeric_value(column, default=0):
        if column is None:
            return default

        value = row[column]

        if pd.isna(value):
            return default

        try:
            return float(value)
        except Exception:
            return default

    pclass = numeric_value(pclass_column, 3)
    age = numeric_value(age_column, 30)
    sibsp = numeric_value(sibsp_column, 0)
    parch = numeric_value(parch_column, 0)
    fare = numeric_value(fare_column, 0)

    sex = "female"
    if sex_column is not None:
        value = str(row[sex_column]).strip().lower()
        if value == "male":
            sex = "male"

    embarked = "S"
    if embarked_column is not None:
        value = str(row[embarked_column]).strip().upper()
        if value in ["S", "C", "Q"]:
            embarked = value

    passenger = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Sex_male": [1 if sex == "male" else 0],
        "Embarked_Q": [1 if embarked == "Q" else 0],
        "Embarked_S": [1 if embarked == "S" else 0]
    })

    passenger = passenger.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return passenger

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_prediction(passenger):
    prediction = int(model.predict(passenger)[0])

    probability = None

    try:
        probabilities = model.predict_proba(passenger)[0]
        probability = float(probabilities[prediction])
    except Exception:
        pass

    return prediction, probability

# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        '<div class="main-title">🚢 Titanic Survival Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Based Survival Prediction System'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1.4, 1])

    with col2:

        st.markdown("""
        <div class="login-card">
            <h2>🔐 Welcome</h2>
            <p>
                Login to access the Titanic passenger
                analytics and prediction system.
            </p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input(
            "Username",
            placeholder="Enter username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password"
        )

        if st.button("🔐 Login", use_container_width=True):

            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.info("Demo Login: admin / 1234")

    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚢 Titanic Survival Prediction")
st.sidebar.caption("Machine Learning Prediction System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👥 Passenger Explorer",
        "🔮 Survival Prediction",
        "📊 Model Performance",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()
st.sidebar.write("Logged in as **admin**")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="page-title">🚢 Titanic Survival Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore historical passenger data and Machine Learning predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero-card">
        <h2>Welcome to the Titanic Analytics System</h2>
        <p>
            This application uses a trained Machine Learning model
            to predict passenger survival based on passenger and
            travel characteristics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total_records = (
        len(titanic_data)
        if titanic_data is not None
        else 891
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Passenger Records", f"{total_records:,}")

    with col2:
        st.metric("Model Accuracy", "80.45%")

    with col3:
        st.metric("Input Features", "8")

    with col4:
        st.metric("Final Model", "Logistic Regression")

    st.markdown(
        '<div class="section-title">📁 Dataset Status</div>',
        unsafe_allow_html=True
    )

    if titanic_data is not None:
        st.success(
            f"Real Titanic dataset loaded successfully: {dataset_path}"
        )
    else:
        st.warning(
            "Titanic CSV was not found. Add the dataset to the data "
            "folder to enable real passenger exploration."
        )

    st.markdown(
        '<div class="section-title">⚙️ How the System Works</div>',
        unsafe_allow_html=True
    )

    steps_col1, steps_col2, steps_col3 = st.columns(3)

    with steps_col1:
        st.info("""
        **1️⃣ Historical Data**

        The system reads real Titanic passenger records
        containing passenger details and known outcomes.
        """)

    with steps_col2:
        st.info("""
        **2️⃣ Machine Learning**

        Passenger features are provided to the trained
        Logistic Regression model.
        """)

    with steps_col3:
        st.info("""
        **3️⃣ Prediction**

        The model predicts class 0 or class 1,
        representing non-survival or survival.
        """)

    st.markdown(
        '<div class="section-title">🚀 Quick Start</div>',
        unsafe_allow_html=True
    )

    st.write("""
    **👥 Passenger Explorer** → Browse real Titanic passengers.

    **🔮 Survival Prediction** → Select a historical passenger
    or enter a custom passenger.

    **📊 Model Performance** → View model accuracy and evaluation.
    """)

# ============================================================
# PASSENGER EXPLORER
# ============================================================

elif page == "👥 Passenger Explorer":

    st.markdown(
        '<div class="page-title">👥 Passenger Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore real passenger records from the Titanic dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    if titanic_data is None:
        st.error("Titanic dataset not found.")
        st.info("Place your Titanic CSV inside the data folder.")
        st.stop()

    name_column = find_column(titanic_data, ["Name"])

    if name_column is not None:

        search = st.text_input(
            "🔎 Search passenger by name",
            placeholder="Example: Braund"
        )

        if search.strip():
            filtered_data = titanic_data[
                titanic_data[name_column]
                .astype(str)
                .str.contains(search, case=False, na=False)
            ]
        else:
            filtered_data = titanic_data

    else:
        filtered_data = titanic_data

    preferred_columns = [
        "PassengerId",
        "Name",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Survived"
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in filtered_data.columns
    ]

    st.dataframe(
        filtered_data[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Showing {len(filtered_data)} passenger record(s)."
    )

# ============================================================
# SURVIVAL PREDICTION
# ============================================================

elif page == "🔮 Survival Prediction":

    st.markdown(
        '<div class="page-title">🔮 Survival Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Use historical passenger data or create a custom passenger profile.'
        '</div>',
        unsafe_allow_html=True
    )

    prediction_mode = st.radio(
        "Choose Prediction Mode",
        ["👤 Historical Passenger", "🧪 Custom Passenger"],
        horizontal=True
    )

    # ========================================================
    # HISTORICAL PASSENGER
    # ========================================================

    if prediction_mode == "👤 Historical Passenger":

        if titanic_data is None:
            st.error("Titanic dataset not found.")
            st.info("Add your Titanic CSV inside the data folder.")
            st.stop()

        name_column = find_column(titanic_data, ["Name"])

        if name_column is None:
            st.error("The dataset does not contain a Name column.")
            st.stop()

        passenger_names = (
            titanic_data[name_column]
            .dropna()
            .astype(str)
            .tolist()
        )

        selected_name = st.selectbox(
            "🔎 Select a real Titanic passenger",
            passenger_names
        )

        selected_rows = titanic_data[
            titanic_data[name_column].astype(str) == selected_name
        ]

        if selected_rows.empty:
            st.error("Passenger record could not be found.")
            st.stop()

        row = selected_rows.iloc[0]

        st.markdown(
            '<div class="section-title">👤 Passenger Profile</div>',
            unsafe_allow_html=True
        )

        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:

            st.markdown(
                '<div class="info-card">',
                unsafe_allow_html=True
            )

            st.markdown(f"### {selected_name}")

            st.write(
                f"**Passenger ID:** "
                f"{get_value(row, ['PassengerId'])}"
            )

            st.write(
                f"**Gender:** {get_value(row, ['Sex'])}"
            )

            st.write(
                f"**Age:** {get_value(row, ['Age'])}"
            )

            st.write(
                f"**Passenger Class:** "
                f"{get_value(row, ['Pclass'])}"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        with profile_col2:

            st.markdown(
                '<div class="info-card">',
                unsafe_allow_html=True
            )

            st.write(
                f"**Ticket:** {get_value(row, ['Ticket'])}"
            )

            st.write(
                f"**Fare:** {get_value(row, ['Fare'])}"
            )

            st.write(
                f"**Cabin:** {get_value(row, ['Cabin'])}"
            )

            st.write(
                f"**Embarked:** {get_value(row, ['Embarked'])}"
            )

            st.write(
                f"**Family:** "
                f"{get_value(row, ['SibSp'], 0)} "
                f"siblings/spouses + "
                f"{get_value(row, ['Parch'], 0)} "
                f"parents/children"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        survived_column = find_column(
            titanic_data,
            ["Survived"]
        )

        actual_survival = None

        if survived_column is not None:
            try:
                actual_survival = int(row[survived_column])
            except Exception:
                pass

        if st.button(
            "🤖 Analyze This Passenger",
            use_container_width=True
        ):

            passenger = create_model_input_from_row(row)
            prediction, probability = make_prediction(passenger)

            st.markdown(
                '<div class="section-title">'
                '🤖 Machine Learning Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:

                st.markdown("""
                <div class="result-card">
                    <div class="result-title">
                        🚢 PREDICTED TO SURVIVE
                    </div>
                    <div class="result-text">
                        The Machine Learning model predicts survival
                        for this passenger.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-card">
                    <div class="result-title">
                        ❌ PREDICTED NOT TO SURVIVE
                    </div>
                    <div class="result-text">
                        The Machine Learning model predicts
                        non-survival for this passenger.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Model Prediction",
                    "SURVIVED" if prediction == 1
                    else "NOT SURVIVED"
                )

            with result_col2:

                if actual_survival is not None:
                    st.metric(
                        "Historical Outcome",
                        "SURVIVED" if actual_survival == 1
                        else "NOT SURVIVED"
                    )
                else:
                    st.metric(
                        "Historical Outcome",
                        "Unavailable"
                    )

            with result_col3:

                if probability is not None:
                    st.metric(
                        "Model Confidence",
                        f"{probability * 100:.1f}%"
                    )
                else:
                    st.metric(
                        "Model Confidence",
                        "Unavailable"
                    )

            if actual_survival is not None:

                if prediction == actual_survival:
                    st.success(
                        "✅ The Machine Learning prediction matches "
                        "the known historical outcome."
                    )
                else:
                    st.warning(
                        "⚠️ The Machine Learning prediction does not "
                        "match the known historical outcome."
                    )

            st.info(
                "The historical outcome is the known value in the "
                "dataset. It is NOT provided to the model as an input."
            )

            with st.expander(
                "🔍 View Features Sent to the Model"
            ):
                st.dataframe(
                    passenger,
                    use_container_width=True,
                    hide_index=True
                )

    # ========================================================
    # CUSTOM PASSENGER
    # ========================================================

    else:

        st.markdown(
            '<div class="section-title">'
            '🧪 Create a Custom Passenger'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Enter passenger characteristics. Do not enter a survival "
            "value—the Machine Learning model will predict it."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            pclass = st.selectbox(
                "Passenger Class",
                [1, 2, 3],
                index=2
            )

        with col2:
            sex = st.selectbox(
                "Gender",
                ["female", "male"]
            )

        with col3:
            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0
            )

        st.markdown(
            '<div class="section-title">'
            '👨‍👩‍👧 Family Information'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            sibsp = st.number_input(
                "Siblings / Spouses Aboard",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )

        with col2:
            parch = st.number_input(
                "Parents / Children Aboard",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )

        st.markdown(
            '<div class="section-title">'
            '🎫 Travel Information'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fare = st.number_input(
                "Ticket Fare",
                min_value=0.0,
                value=30.0,
                step=1.0
            )

        with col2:
            embarked = st.selectbox(
                "Port of Embarkation",
                ["S", "C", "Q"]
            )

        if st.button(
            "🚢 Predict Custom Passenger",
            use_container_width=True
        ):

            passenger = pd.DataFrame({
                "Pclass": [pclass],
                "Age": [age],
                "SibSp": [sibsp],
                "Parch": [parch],
                "Fare": [fare],
                "Sex_male": [1 if sex == "male" else 0],
                "Embarked_Q": [1 if embarked == "Q" else 0],
                "Embarked_S": [1 if embarked == "S" else 0]
            })

            passenger = passenger.reindex(
                columns=feature_columns,
                fill_value=0
            )

            prediction, probability = make_prediction(passenger)

            if prediction == 1:

                st.markdown("""
                <div class="result-card">
                    <div class="result-title">
                        🚢 SURVIVAL PREDICTED
                    </div>
                    <div class="result-text">
                        The model predicts that this passenger
                        belongs to the survived class.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-card">
                    <div class="result-title">
                        ❌ NON-SURVIVAL PREDICTED
                    </div>
                    <div class="result-text">
                        The model predicts that this passenger
                        belongs to the non-survived class.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Prediction",
                    "SURVIVED" if prediction == 1
                    else "NOT SURVIVED"
                )

            with col2:

                if probability is not None:
                    st.metric(
                        "Model Confidence",
                        f"{probability * 100:.1f}%"
                    )
                else:
                    st.metric(
                        "Model Confidence",
                        "Unavailable"
                    )

            with st.expander("🔍 View Model Input"):
                st.dataframe(
                    passenger,
                    use_container_width=True,
                    hide_index=True
                )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="page-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Evaluation and information about the Machine Learning models.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🤖 Model Comparison</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🏆 Logistic Regression</h3>
            <h1>80.45%</h1>
            <p>Selected Final Model</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🌳 Decision Tree</h3>
            <h1>76.54%</h1>
            <p>Comparison Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        '📋 Dataset Information'
        '</div>',
        unsafe_allow_html=True
    )

    total_records = (
        len(titanic_data)
        if titanic_data is not None
        else 891
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{total_records:,}")

    with col2:
        st.metric("Training Records", "712")

    with col3:
        st.metric("Testing Records", "179")

    st.markdown(
        '<div class="section-title">'
        '📌 Features Used by Final Model'
        '</div>',
        unsafe_allow_html=True
    )

    features = pd.DataFrame({
        "Feature": [
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex_male",
            "Embarked_Q",
            "Embarked_S"
        ],
        "Meaning": [
            "Passenger class",
            "Passenger age",
            "Siblings / spouses aboard",
            "Parents / children aboard",
            "Ticket fare",
            "Male gender indicator",
            "Queenstown indicator",
            "Southampton indicator"
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">'
        '🧠 How Prediction Works'
        '</div>',
        unsafe_allow_html=True
    )

    st.info("""
    The model was trained using historical Titanic records where
    the actual survival outcome was known.

    During prediction, the survival value is NOT provided to the model.

    The model receives passenger characteristics and predicts:

    **0 → Did Not Survive**

    **1 → Survived**
    """)

    st.success(
        "Logistic Regression achieved 80.45% test accuracy "
        "and was selected as the final model."
    )

    st.warning(
        "Predictions are Machine Learning estimates and are "
        "not guaranteed real-world outcomes."
    )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="page-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Titanic Survival Prediction — Industrial Training Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>🎯 Project Objective</h3>
        <p>
        The objective of this project is to develop a Machine Learning
        classification system that predicts Titanic passenger survival
        using passenger and travel characteristics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">'
        '🛠️ Technologies Used'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("🐍 **Python**\n\nProgramming")

    with col2:
        st.info("🐼 **Pandas**\n\nData Processing")

    with col3:
        st.info("🤖 **Scikit-learn**\n\nMachine Learning")

    with col4:
        st.info("🌐 **Streamlit**\n\nWeb Application")

    st.markdown(
        '<div class="section-title">🔄 Project Flow</div>',
        unsafe_allow_html=True
    )

    st.write("""
    **Historical Dataset**

    ↓

    **Data Preprocessing**

    ↓

    **Feature Engineering**

    ↓

    **Model Training**

    ↓

    **Model Evaluation**

    ↓

    **Best Model Selection**

    ↓

    **Model Saving**

    ↓

    **Streamlit Application**

    ↓

    **Passenger Prediction**
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    🚢 Titanic Survival Prediction |
    Machine Learning Industrial Training Project
</div>
""", unsafe_allow_html=True)
