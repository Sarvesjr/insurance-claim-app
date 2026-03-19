import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import train_claim_approval_model, train_claim_amount_model, predict_claim
from utils import process_damage_image, classify_damage_severity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Insurance Analyzer", layout="wide")

# -----------------------------
# CLEAN PREMIUM UI
# -----------------------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Segoe UI', sans-serif;
}

/* HEADINGS */
h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #38bdf8;
}

h2, h3 {
    color: #7dd3fc;
}

/* REMOVE EXTRA BOX */
.block-container > div {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}

/* GLASS CARD */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
    transition: 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(56,189,248,0.15);
}

/* INPUTS */
.stTextInput input,
.stNumberInput input,
.stSelectbox div {
    background-color: rgba(2,6,23,0.9) !important;
    color: white !important;
    border-radius: 8px;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.05);
}

/* METRICS */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# CLAIM CALCULATION
# -----------------------------
def calculate_claim(damage_percent, car_value, car_age, insurance_type):
    repair_cost = (damage_percent / 100) * car_value
    depreciation = min(car_age * 0.05, 0.5)
    coverage = 0.8 if insurance_type == "Comprehensive" else 0.5
    deductible = 5000

    claim_amount = repair_cost * (1 - depreciation) * coverage - deductible
    claim_amount = max(claim_amount, 0)

    user_pay = repair_cost - claim_amount

    return repair_cost, claim_amount, user_pay, depreciation, coverage

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    return train_claim_approval_model(), train_claim_amount_model()

approval_model, amount_model = load_models()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1>AI Car Insurance Claim Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>AI-powered damage detection and claim estimation system</p>", unsafe_allow_html=True)

# -----------------------------
# USER INFO
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.subheader("User Information")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Full Name")
    age = st.number_input("Age", 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    city = st.selectbox("City", ["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad"])
    car_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "Luxury"])
    car_age = st.number_input("Car Age (Years)", 0, 20, 3)

with col3:
    insurance_type = st.selectbox("Insurance Type", ["Third-party", "Comprehensive"])
    previous_claims = st.selectbox("Previous Claims", [0, 1, 2, 3])
    car_value = st.number_input("Car Value (INR)", 100000, 10000000, 500000)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# VEHICLE DETAILS
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.subheader("Vehicle Details")

col4, col5 = st.columns(2)

with col4:
    car_brand = st.selectbox(
        "Car Brand",
        ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Toyota",
         "Honda", "Kia", "MG", "Skoda", "Volkswagen", "BMW", "Mercedes",
         "Audi", "Volvo", "Jaguar", "Land Rover", "Porsche", "Other"]
    )

with col5:
    if car_brand == "Other":
        car_brand_custom = st.text_input("Enter Custom Brand")
    else:
        car_brand_custom = car_brand

car_name = st.text_input("Car Model Name")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.subheader("Upload Damage Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# PROCESS IMAGE
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", width='stretch')

    processed_img, damage_percent = process_damage_image(img_array)
    severity = classify_damage_severity(damage_percent)

    with col2:
        st.image(processed_img, caption="Processed Image", width='stretch')

    st.markdown('</div>', unsafe_allow_html=True)

    # DAMAGE SUMMARY
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("Damage Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Damage Percentage", f"{damage_percent:.1f}%")
    c2.metric("Severity", severity)
    c3.metric("Car Value", f"₹{car_value:,.0f}")

    st.markdown(f"Vehicle: {car_brand_custom} {car_name} ({car_type})")
    st.progress(min(damage_percent / 100, 1.0))

    st.markdown('</div>', unsafe_allow_html=True)

    # PREDICT
    if st.button("Predict Claim", width='stretch'):

        if not name or not car_name:
            st.error("Please fill all required fields")
        else:
            insurance_encoded = 1 if insurance_type == "Comprehensive" else 0

            is_approved, _ = predict_claim(
                approval_model,
                amount_model,
                age,
                car_age,
                car_value,
                previous_claims,
                damage_percent,
                insurance_encoded
            )

            repair_cost, claim_amount, user_pay, depreciation, coverage = calculate_claim(
                damage_percent,
                car_value,
                car_age,
                insurance_type
            )

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            st.subheader("Claim Breakdown")

            c1, c2, c3 = st.columns(3)
            c1.metric("Repair Cost", f"₹{repair_cost:,.0f}")
            c2.metric("Insurance Pays", f"₹{claim_amount:,.0f}")
            c3.metric("You Pay", f"₹{user_pay:,.0f}")

            if is_approved:
                st.success("Claim Approved based on AI model")
            else:
                st.error("Claim Rejected based on AI model")

            st.markdown('</div>', unsafe_allow_html=True)

            # EXPLANATION
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            st.subheader("Detailed Explanation")

            st.markdown(f"""
Policyholder Details:
- Name: {name}
- Age: {age}
- City: {city}

Vehicle Information:
- Type: {car_type}
- Model: {car_brand_custom} {car_name}
- Age: {car_age} years
- Value: ₹{car_value:,.0f}

Damage Assessment:
- Damage Percentage: {damage_percent:.1f}%
- Severity: {severity}

Financial Calculation:
- Estimated Repair Cost: ₹{repair_cost:,.0f}
- Depreciation Applied: {depreciation*100:.0f}%
- Coverage Applied: {coverage*100:.0f}%
- Deductible: ₹5,000

Final Outcome:
- Insurance Pays: ₹{claim_amount:,.0f}
- Customer Pays: ₹{user_pay:,.0f}
""")

            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Please upload an image to begin analysis")