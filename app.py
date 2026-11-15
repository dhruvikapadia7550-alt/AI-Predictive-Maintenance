import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Main Heading
st.title("⚙️ AI Predictive Maintenance System")

st.write(
    "Predict machine failure using Machine Learning "
    "based on machine operating conditions."
)

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Home", "Prediction", "About"]
)
if page == "Dashboard":

    import pandas as pd
    import plotly.express as px

    st.title("🏭 Industrial Predictive Maintenance Dashboard")

    st.write(
        "Monitor machine performance and analyze potential equipment failures."
    )

    # Load dataset
    data = pd.read_csv("dataset/ai4i2020.csv")

    # Calculate statistics
    total_machines = len(data)

    failed_machines = data["Machine failure"].sum()

    normal_machines = total_machines - failed_machines

    failure_percentage = (failed_machines / total_machines) * 100

    
    # Professional Dashboard Metrics

    st.subheader("Machine Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🏭 Total Machines",
        f"{total_machines:,}",
        help="Total machine records in the dataset"
    )

    col2.metric(
        "✅ Normal Machines",
        f"{normal_machines:,}",
        help="Records without machine failure"
    )

    col3.metric(
        "⚠️ Failed Machines",
        f"{failed_machines:,}",
        help="Records with machine failure"
    )

    col4.metric(
        "📊 Failure Rate",
        f"{failure_percentage:.2f}%",
        help="Percentage of records with machine failure"
    )

    st.divider()



    st.divider()

    # Machine Failure Distribution
    st.subheader("Machine Failure Distribution")

    failure_data = pd.DataFrame({
        "Machine Status": ["Normal", "Failure"],
        "Count": [normal_machines, failed_machines]
    })

    fig = px.pie(
        failure_data,
        names="Machine Status",
        values="Count",
        title="Normal vs Failed Machine Records",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Failure Analysis
    st.subheader("Failure Analysis")

    failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    failure_counts = data[failure_types].sum().reset_index()

    failure_counts.columns = ["Failure Type", "Count"]

    fig2 = px.bar(
        failure_counts,
        x="Failure Type",
        y="Count",
        title="Distribution of Different Failure Types",
        text="Count"
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.divider()

    # Machine Parameter Analysis
    st.subheader("Machine Parameter Analysis")

    parameter = st.selectbox(
        "Select Machine Parameter",
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )

    fig3 = px.box(
        data,
        x="Machine failure",
        y=parameter,
        color="Machine failure",
        title=f"{parameter} vs Machine Failure",
        labels={
            "Machine failure": "Failure Status",
            parameter: parameter
        }
    )

    st.plotly_chart(fig3, use_container_width=True)
    st.divider()
    st.divider()

    # Machine Learning Model Comparison

    st.subheader("🤖 Machine Learning Model Comparison")

    comparison_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_comparison_results.csv")

    if os.path.exists(comparison_file):

        comparison_data = pd.read_csv(comparison_file)

        st.write(
            "Performance comparison of three machine learning algorithms."
        )

       # Display comparison table without PyArrow
        st.subheader("Model Performance Results")

        display_data = comparison_data.copy()

        for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            if column in display_data.columns:
                display_data[column] = display_data[column].apply(
                    lambda value: f"{value:.2%}" if pd.notna(value) else "-"
                )

        st.markdown(
            display_data.to_html(index=False, border=0),
            unsafe_allow_html=True
        )

        # Prepare chart data
        chart_data = comparison_data.melt(
            id_vars="Model",
            value_vars=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            var_name="Performance Metric",
            value_name="Score"
        )

        # Create comparison chart
        fig4 = px.bar(
            chart_data,
            x="Model",
            y="Score",
            color="Performance Metric",
            barmode="group",
            title="Performance Comparison of ML Algorithms",
            text_auto=".2%"
        )

        fig4.update_layout(
            yaxis=dict(range=[0, 1]),
            yaxis_title="Performance Score",
            xaxis_title="Machine Learning Algorithm"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    else:   

        st.warning(
            "Model comparison results not found. "
            "Please run model_comparison.py first."
        )
    st.divider()

    # Random Forest Confusion Matrix

    st.subheader("🎯 Random Forest Confusion Matrix")

    confusion_file = "confusion_matrix.csv"

    if os.path.exists(confusion_file):
        cm_data = pd.read_csv(confusion_file, index_col=0)

        fig_cm = px.imshow(
            cm_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Actual vs Predicted Machine Status",
            labels={
                "x": "Predicted Status",
                "y": "Actual Status",
                "color": "Number of Records"
            }
        )

        fig_cm.update_layout(
            height=600,
            margin=dict(l=100, r=60, t=100, b=100),
            font=dict(size=16),
            title_font_size=22
        )

        fig_cm.update_traces(
            textfont_size=20
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

    else:
        st.warning("Confusion matrix file not found.")


    # Prediction History
    st.subheader("📋 Machine Prediction History")

    history_file = "prediction_history.csv"

    if os.path.exists(history_file):

        history = pd.read_csv(history_file)

        st.dataframe(
            history,
            use_container_width=True
        )

        st.download_button(
            label="Download Prediction History",
            data=history.to_csv(index=False),
            file_name="machine_prediction_history.csv",
            mime="text/csv"
        )

    else:

        st.info("No prediction history available yet.")
# Home Page
if page == "Home":

    st.header("Welcome to Predictive Maintenance")

    st.write("""
    This application uses Machine Learning to predict
    whether a machine is likely to fail based on
    its operating parameters.

    The system helps identify potential failures
    before they occur.
    """)

    st.success("Machine Learning Model: Random Forest")

    st.info("Model Accuracy: 97.15%")

elif page == "Prediction":

    st.title("⚙️ Machine Failure Prediction")

    st.write("Enter machine parameters to predict potential machine failure.")

    # Machine Type
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    # Machine Parameters
    air_temperature = st.number_input(
        "Air Temperature (K)",
        value=300.0
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        value=310.0
    )

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        value=100
    )

    # Prediction Button
    if st.button("Predict Machine Failure"):

        import pandas as pd
        import joblib

        # Load trained model
        model = joblib.load("src/maintenance_model.pkl")

        # Convert machine type
        type_mapping = {"L": 0, "M": 1, "H": 2}

        # Prepare input
        input_data = pd.DataFrame({
            "Type": [type_mapping[machine_type]],
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear]
        })

        # Make prediction
        prediction = model.predict(input_data)
                # Calculate prediction probabilities
        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)

        failure_probability = probabilities[classes.index(1)] * 100

        normal_probability = probabilities[classes.index(0)] * 100
        import os
        from datetime import datetime

        # Convert prediction into readable status
        prediction_label = (
            "Failure" if prediction[0] == 1 else "Normal"
        )

        # Create prediction history record
        history_file = "prediction_history.csv"

        new_record = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Machine Type": [machine_type],
            "Air Temperature": [air_temperature],
            "Process Temperature": [process_temperature],
            "Rotational Speed": [rotational_speed],
            "Torque": [torque],
            "Tool Wear": [tool_wear],
            "Prediction": [prediction_label]
        })

        # Save record in CSV
        if os.path.exists(history_file):
            new_record.to_csv(
                history_file,
                mode="a",
                header=False,
                index=False
            )
        else:
            new_record.to_csv(
                history_file,
                index=False
            )

        st.success("Prediction saved to history!")
        # Display result
        if prediction[0] == 1:

            st.error("⚠️ Machine Failure Predicted!")

            st.warning(
                "Potential failure detected. "
                "Maintenance inspection is recommended."
            )

        else:

            st.success("✅ No Machine Failure Predicted")

            st.info(
                "The machine is predicted to be operating normally."
            )
                    # Prediction Confidence

        st.divider()

        st.subheader("📊 Prediction Confidence")

        col1, col2 = st.columns(2)

        col1.metric(
            "Failure Probability",
            f"{failure_probability:.2f}%"
        )

        col2.metric(
            "Normal Probability",
            f"{normal_probability:.2f}%"
        )

        st.progress(int(failure_probability))
                    # Maintenance Recommendations

        st.divider()

        st.subheader("🔧 Maintenance Recommendations")

        recommendations = []

        if tool_wear > 200:
            recommendations.append(
                "High tool wear detected. Inspect and consider replacing the tool."
            )

        if torque > 60:
            recommendations.append(
                "High torque detected. Inspect machine load and mechanical components."
            )

        if air_temperature > 305:
            recommendations.append(
                "High air temperature detected. Inspect the cooling and ventilation system."
            )

        if process_temperature > 315:
            recommendations.append(
                "High process temperature detected. Check the temperature control system."
            )

        if prediction[0] == 1:
            recommendations.append(
                "Machine failure predicted. Arrange a maintenance inspection."
            )

        if recommendations:

            for recommendation in recommendations:
                st.warning("⚠️ " + recommendation)

        else:

            st.success(
                "✅ No immediate maintenance concerns identified by these basic checks."
            )

# About Page
elif page == "About":

    st.header("About the Project")

    st.write("""
    Project Name: AI Predictive Maintenance

    Technology Used:
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib
    - Machine Learning
    """)
