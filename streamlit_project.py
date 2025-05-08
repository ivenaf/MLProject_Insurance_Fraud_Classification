import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with wide layout and custom theme
st.set_page_config(layout="wide")

# Custom CSS to style the sidebar with a blue gradient and improve radio buttons
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#2C3E50, #4CA1AF);
        color: white;
    }
    
    /* Increase radio button label size and change color */
    [data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF;
        font-size: 18px !important;
        font-weight: 600;
    }
    
    /* Make the "Go to:" text larger and more visible */
    .sidebar .sidebar-content label {
        color: #FFFFFF !important;
        font-size: 20px !important;
        font-weight: bold;
    }
    
    /* Make all sidebar text more visible */
    [data-testid="stSidebar"] p {
        color: #FFFFFF;
        font-size: 18px !important;
    }
    
    [data-testid="stSidebarNav"] {
        background-color: rgba(0, 0, 0, 0);
    }
    </style>
""", unsafe_allow_html=True)

df=pd.read_csv("insurance_claims.csv")
X_train=pd.read_csv("/opt/anaconda3/envs/Env1/claim_predict/X_train.csv")

st.title("Insurance Claim Prediction ")
st.sidebar.title("Table of contents")
pages=["Business problem", "Data Visualization","Statistical Methods", "Feature Engineering",  "Modelling", "Real-life Sample Application"]
st.sidebar.write("\n\nCreated by:")
st.sidebar.write("Bertrand Tcheuffa  \n Nathalie Mugrauer  \n Quy-Manh Jurca-Tsan \n")
st.sidebar.write("\n\n\n")
# Add a more visible label before the radio buttons
st.sidebar.markdown("<p style='font-size:20px; font-weight:bold; color:white;'>Go to:</p>", unsafe_allow_html=True)
page=st.sidebar.radio("", pages, label_visibility="collapsed")  # Hide the default label
  

#####################################################################################################################

## Feature Engineering
if page == pages[3] :
  st.write("### Feature Engineering")
  st.write("In the previous slides we have gained a solid understanding of our data. Based on statistical methods we have reduced the dimensionality of our dataset from 40 explanatory features to 8  **(7 categorical and 1 quantitative)**." )
  #show X_train.info()
  st.image("df.info().png")
  
  st.write("\nMissing values in **'authorities_contacted'** were handled using the **SimpleImputer()** with the **strategy 'most_frequent'**.")
  st.code("""categorical_columns = ['authorities_contacted'] 
          categorical_imputer=SimpleImputer(strategy='most_frequent')

 X_train[categorical_columns]=categorical_imputer.fit_transform(X_train[categorical_columns])
 X_test[categorical_columns]=categorical_imputer.transform(X_test[categorical_columns])
 """)

 #categorical features
  st.write("\nConsequently, we reviewed these features and decided if and which transformations are needed. We begin with the categorical features:")
  st.image("cat_var.png")
  #shape of X_train
  st.write("/nThe shape of the transformed TRAIN Set is:", X_train.shape)
  
  st.dataframe(X_train.head(3))
 #quantitative features
  st.write("\nThe only quantitative feature is **'vehicle_claim'**.  We observe a wide range among the min and max values.  We decided to scale this feature using the **StandardScaler()**.")
  
  st.code("""
minmax_scaler = MinMaxScaler()
minmax_col = ['vehicle_claim']          
for column in minmax_col:
    X_train[column] = minmax_scaler.fit_transform(X_train[[column]])
    X_test[column] = minmax_scaler.transform(X_test[[column]])
  """)
  
  st.write("**Dataframe  before scaling:**")
  vehicle_claim_desc = df['vehicle_claim'].describe()[['count', 'mean', 'min', 'max']]
  vehicle_claim_desc_df = vehicle_claim_desc.to_frame().T
  st.dataframe(vehicle_claim_desc_df)

  st.write("**Train set after scaling:**")
  vehicle_claim_desc_X_train = X_train['vehicle_claim'].describe()[['count', 'mean', 'min', 'max']].to_frame().T
  st.dataframe(vehicle_claim_desc_X_train)

################################################################################################
################################################################################################

# --- 4. Real-life Sample Application ---
if page == pages[5]:
    st.markdown("<h2 style='color:red;'>Real-life Sample Application: Insurance Fraud Detection</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:red;'>This application demonstrates how the machine learning model could be used in a real-world scenario to assess the risk of fraud.</p>", unsafe_allow_html=True)

    # --- 4.1.  Application Scenario: New Insurance Policy ---
    def new_policy_survey():
        st.subheader("New Insurance Policy Application")
        st.write("Please provide the following information to process your insurance application.")

        # Use a dictionary to structure the questions and their input types
        questions = {
            "Incident Severity": st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]),
            "Incident State": st.selectbox("Incident State", df['incident_state'].unique()),
            "Insured Hobbies": st.multiselect("Insured Hobbies", df['insured_hobbies'].unique()),
            "Collision Type": st.selectbox("Collision Type", ["No Collision", "Rear Collision", "Front Collision", "Side Collision", "Other"]),
            "Property Damage": st.radio("Property Damage", ["YES", "NO", "Unknown"]),
            "Authorities Contacted": st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"]),
            "Vehicle Claim Amount": st.number_input("Vehicle Claim Amount", min_value=0.0, format="%.2f"),
            "Incident Type": st.selectbox("Incident Type", df['incident_type'].unique()),
        }

        # Store the answers in a dictionary
        answers = {question: answer for question, answer in questions.items()}

        # Convert the answers to a DataFrame (for consistency)
        answers_df = pd.DataFrame([answers])

        # Display the answers
        st.subheader("Your Answers:")
        st.dataframe(answers_df)

        # "Submit" button (no actual model prediction here, just for show)
        if st.button("Submit Application"):
            st.success("Your application has been submitted.  A representative will contact you.")
            # In a real app, you'd send 'answers_df' to your fraud prediction model here


    # --- 4.2. Claim Scenario:  Filing an Insurance Claim ---
    def file_claim_survey():
        st.subheader("File an Insurance Claim")
        st.write("Please provide details about the incident to process your claim.")

        questions = {
            "Incident Severity": st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]),
            "Incident State": st.selectbox("Incident State", df['incident_state'].unique()),
             "Insured Hobbies": st.multiselect("Insured Hobbies", df['insured_hobbies'].unique()),
            "Collision Type": st.selectbox("Collision Type", ["No Collision", "Rear Collision", "Front Collision", "Side Collision", "Other"]),
            "Property Damage": st.radio("Property Damage", ["YES", "NO", "Unknown"]),
            "Authorities Contacted": st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"]),
            "Vehicle Claim Amount": st.number_input("Vehicle Claim Amount", min_value=0.0, format="%.2f"),
            "Incident Type": st.selectbox("Incident Type", df['incident_type'].unique()),
            "Description of Incident": st.text_area("Description of Incident", height=100),  # Added description
            "Date of Incident": st.date_input("Date of Incident"),  # Added date
            "Location of Incident": st.text_input("Location of Incident"), #Added Location
        }
        answers = {question: answer for question, answer in questions.items()}
        answers_df = pd.DataFrame([answers])

        st.subheader("Your Claim Details:")
        st.dataframe(answers_df)

        if st.button("Submit Claim"):
            st.success("Your claim has been submitted. We will contact you with an update.")
            #  send 'answers_df' to your fraud prediction model


    # --- 4.3.  Main App  ---
    def main():
        # st.title("Insurance Fraud Detection") #moved to main page
        st.sidebar.title("Choose a Scenario")
        scenario = st.sidebar.radio("Select a Scenario", ["New Policy Application", "File a Claim"])

        if scenario == "New Policy Application":
            new_policy_survey()
        elif scenario == "File a Claim":
            file_claim_survey()

    if __name__ == "__main__":
        main()
