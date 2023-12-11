import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle 


# Placeholder function for prediction and interpretation
def predict_activity(acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, wrist):
    with open("StandardScalar.pkl", 'rb') as file:
        Scaling = pickle.load(file)
    Scaled_data=Scaling.transform([[acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, wrist]])
        
    with open("random_forest_model.pkl", 'rb') as file:
        data = pickle.load(file)
        
    prediction = data.predict(Scaled_data)
    return prediction
    

def interpret_prediction(prediction):
    return "Walking" if prediction == 0 else "Running"

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable Matplotlib warning

    html_temp = """
    <div style="background-color:tomato;padding:10px;text-align:center;">
    <h2 style="color:white;">Walk Run Classification App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    acceleration_x = st.sidebar.number_input("Enter acceleration_x", format="%f", value=0.0)
    acceleration_y = st.sidebar.number_input("Enter acceleration_y", format="%f", value=0.0)
    acceleration_z = st.sidebar.number_input("Enter acceleration_z", format="%f", value=0.0)
    gyro_x = st.sidebar.number_input("Enter gyro_x", format="%f", value=0.0)
    gyro_y = st.sidebar.number_input("Enter gyro_y", format="%f", value=0.0)
    gyro_z = st.sidebar.number_input("Enter gyro_z", format="%f", value=0.0)
    wrist = st.sidebar.number_input("Enter wrist", format="%f", value=0.0)

    # Prediction button
    if st.sidebar.button("Predict"):
        prediction = predict_activity(acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z, wrist)
        result = interpret_prediction(prediction)
        st.sidebar.success(f" The Person is : {result}")

    # Middle section for project info
    st.subheader("Project Overview 🚀")

    
    st.write("🚀 Welcome to the Walk Run Classification project, where technology meets motion!")

    st.write("🏃‍♂️🚶‍♀️ In this cutting-edge endeavor, we leverage the power of deep learning and machine learning models to decode the subtle dance of human movement. Our algorithms can distinguish between a leisurely stroll and a brisk run, all from raw accelerometer and gyroscope data.")


    st.write("👟 So, lace up your virtual sneakers and embark on this journey with us. Let's explore the fascinating world where every data point tells a story, and every prediction captures a unique rhythm in the symphony of human locomotion. Happy predicting! 🌟")

    # Dynamic line plot showing variation of activity against features
    st.subheader("Unleashing the Dance of Data 🕺💃")
    features = ["acceleration_x", "acceleration_y", "acceleration_z", "gyro_x", "gyro_y", "gyro_z", "wrist"]
    activity_values = np.random.rand(len(features))  # Replace with actual values
    plt.plot(features, activity_values, marker='o')
    plt.xlabel("Features")
    plt.ylabel("Activity Value")
    plt.title("Activity Variation Against Features")

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')  # Adjust the rotation angle as needed

    st.pyplot(plt)

    
    
    # Contact section
    st.subheader("Contact 📧")

    st.write("Have questions or suggestions? Feel free to reach out to us at [Ketkishinde2904@gmail.com]")
    
    
if __name__ == '__main__':
    main()
