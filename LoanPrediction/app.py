import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: #F4F7FB;
}

/* Main container */
.block-container {
    max-width: 1250px;
    padding-top: 25px;
    padding-bottom: 40px;
}

/* Headings */
h1, h2, h3 {
    color: #172554 !important;
}

/* Normal text */
p, label {
    color: #1E293B !important;
}

/* Header */
.header {
    background: linear-gradient(
        135deg,
        #172554,
        #4338CA
    );

    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
}

.header-title {
    color: white !important;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.header-subtitle {
    color: #E0E7FF !important;
    font-size: 18px;
}

/* Cards */
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #E2E8F0;
    margin-bottom: 20px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #172554 !important;
}

/* Inputs */
input {
    background-color: white !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 1px solid #CBD5E1 !important;
}

div[data-baseweb="select"] * {
    color: #172033 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    min-height: 58px;

    background: linear-gradient(
        135deg,
        #4F46E5,
        #2563EB
    ) !important;

    color: white !important;

    border: none !important;
    border-radius: 15px !important;

    font-size: 17px !important;
    font-weight: 800 !important;
}

.stButton > button:hover {
    background: #3730A3 !important;
    color: white !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #334155 !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #4338CA !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 15px !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load("loan_model.pkl")


try:

    model = load_model()

except Exception as e:

    st.error("❌ Loan model could not be loaded.")

    st.code(str(e))

    st.stop()


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="header">

<div class="header-title">
🏦 Loan Eligibility Prediction System
</div>

<div class="header-subtitle">
Machine Learning Based Loan Approval Prediction
</div>

</div>
""", unsafe_allow_html=True)


# =====================================================
# DASHBOARD METRICS
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "⚡ Prediction",
        "Instant"
    )

with c2:

    st.metric(
        "🤖 Model",
        "Logistic Regression"
    )

with c3:

    st.metric(
        "📊 Features",
        "10"
    )

with c4:

    st.metric(
        "💰 Loan Range",
        "₹50K - ₹1Cr"
    )


st.write("")


# =====================================================
# INPUT SECTION
# =====================================================

st.header("📋 Applicant Information")

tab1, tab2, tab3 = st.tabs(
    [
        "👤 Personal",
        "💰 Financial",
        "🏠 Property"
    ]
)


# =====================================================
# PERSONAL INFORMATION
# =====================================================

with tab1:

    st.subheader("Personal Details")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Marital Status",
            ["Yes", "No"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

    with col2:

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        credit_history_text = st.selectbox(
            "Credit History",
            [
                "Good",
                "Poor"
            ]
        )


# =====================================================
# FINANCIAL INFORMATION
# =====================================================

with tab2:

    st.subheader("Financial Details")

    col1, col2 = st.columns(2)

    with col1:

        applicant_income = st.number_input(
            "Applicant Monthly Income (₹)",
            min_value=5000,
            max_value=10000000,
            value=50000,
            step=5000
        )

        coapplicant_income = st.number_input(
            "Co-applicant Monthly Income (₹)",
            min_value=0,
            max_value=10000000,
            value=0,
            step=5000
        )

    with col2:

        loan_amount_rupees = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=50000,
            max_value=100000000,
            value=500000,
            step=50000
        )

        loan_term = st.selectbox(
            "Loan Term (Months)",
            [
                12,
                24,
                36,
                48,
                60,
                84,
                120,
                180,
                240,
                300,
                360
            ],
            index=10
        )

    # Display amount
    if loan_amount_rupees >= 10000000:

        display_amount = (
            f"₹{loan_amount_rupees / 10000000:.2f} Crore"
        )

    else:

        display_amount = (
            f"₹{loan_amount_rupees / 100000:.2f} Lakh"
        )

    st.info(
        f"💰 Requested Loan Amount: **{display_amount}**"
    )


# =====================================================
# PROPERTY INFORMATION
# =====================================================

with tab3:

    st.subheader("Property Details")

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semiurban",
            "Rural"
        ]
    )

    st.write("")

    st.info(
        "Property location can influence the model's prediction."
    )


# =====================================================
# CONVERT VALUES FOR MODEL
# =====================================================

credit_history = (
    1
    if credit_history_text == "Good"
    else 0
)


# Dataset commonly stores LoanAmount in thousands
loan_amount_model = loan_amount_rupees / 1000


# =====================================================
# CREATE MODEL INPUT
# =====================================================

input_data = pd.DataFrame({

    "Gender": [gender],

    "Married": [married],

    "Education": [education],

    "Self_Employed": [self_employed],

    "ApplicantIncome": [
        applicant_income
    ],

    "CoapplicantIncome": [
        coapplicant_income
    ],

    "LoanAmount": [
        loan_amount_model
    ],

    "Loan_Amount_Term": [
        loan_term
    ],

    "Credit_History": [
        credit_history
    ],

    "Property_Area": [
        property_area
    ]

})


# =====================================================
# APPLICATION SUMMARY
# =====================================================

st.divider()

st.header("📊 Application Summary")

s1, s2, s3, s4 = st.columns(4)

with s1:

    st.metric(
        "Monthly Income",
        f"₹{applicant_income:,.0f}"
    )

with s2:

    st.metric(
        "Co-applicant Income",
        f"₹{coapplicant_income:,.0f}"
    )

with s3:

    st.metric(
        "Loan Amount",
        f"₹{loan_amount_rupees:,.0f}"
    )

with s4:

    st.metric(
        "Loan Term",
        f"{loan_term} Months"
    )


# =====================================================
# PREDICTION
# =====================================================

st.divider()

st.header("🔮 Loan Eligibility Prediction")

st.write(
    "Click the button below to analyze the applicant information."
)


if st.button(
    "🚀 PREDICT LOAN ELIGIBILITY",
    type="primary"
):

    try:

        # Prediction
        prediction = model.predict(
            input_data
        )[0]

        result = str(
            prediction
        ).strip().upper()


        # =================================================
        # RESULT
        # =================================================

        if result in [
            "Y",
            "YES",
            "1",
            "APPROVED",
            "ELIGIBLE"
        ]:

            st.success(
                "🎉 LOAN ELIGIBLE"
            )

            st.write(
                "The machine learning model predicts that "
                "the applicant is eligible for loan approval."
            )

        else:

            st.warning(
                "⚠️ LOAN NOT ELIGIBLE"
            )

            st.write(
                "The machine learning model predicts that "
                "the applicant may not be eligible for loan approval."
            )


        # =================================================
        # PROBABILITY
        # =================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = max(
                probabilities
            ) * 100


            st.subheader(
                "🤖 Model Confidence"
            )

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )


        # =================================================
        # DATA SENT TO MODEL
        # =================================================

        with st.expander(
            "🔍 View Data Sent to Model"
        ):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(e)
        )

        st.info(
            "Make sure train_model.py was run after "
            "the latest changes."
        )


# =====================================================
# ABOUT
# =====================================================

st.divider()

with st.expander(
    "ℹ️ About This Project"
):

    st.write(
        """
        **Loan Eligibility Prediction System**

        This project uses machine learning to predict
        whether a loan application is likely to be approved
        based on applicant information.

        The classification model uses:

        • Gender  
        • Marital Status  
        • Education  
        • Self Employment  
        • Applicant Income  
        • Co-applicant Income  
        • Loan Amount  
        • Loan Term  
        • Credit History  
        • Property Area  

        **Model:** Logistic Regression

        This is an academic machine-learning project and
        should not be treated as an actual bank approval system.
        """
    )


st.caption(
    "🏦 Loan Eligibility Prediction System | Machine Learning Academic Project"
)