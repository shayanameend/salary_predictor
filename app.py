import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Salary Predictor Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_data
def load_data():
    df = pd.read_csv("output/merged_cleaned_salary_data.csv")
    categorical_cols = ['Country', 'EducationLevel', 'DeveloperType', 'CompanySize', 'RemoteWork']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', 'Unknown')

    return df

@st.cache_resource
def load_models():
    preprocessor = joblib.load("output/preprocessor.joblib")
    model = joblib.load("output/salary_predictor.joblib")
    return preprocessor, model

try:
    data = load_data()
    preprocessor, model = load_models()
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

def predict_salary(country, education, experience, dev_type, company_size, remote_work):

    input_data = pd.DataFrame({
        'Country': [country],
        'EducationLevel': [education],
        'YearsExperience': [float(experience)],
        'ExpSquared': [float(experience) ** 2],
        'DeveloperType': [dev_type],
        'CompanySize': [company_size],
        'RemoteWork': [remote_work]
    })


    input_processed = preprocessor.transform(input_data)


    log_prediction = model.predict(input_processed)[0]


    predicted_salary = np.expm1(log_prediction)


    lower_bound = predicted_salary * 0.8
    upper_bound = predicted_salary * 1.2

    return predicted_salary, lower_bound, upper_bound

def navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Salary Prediction", "Model Insights"])
    return page

def home_page():
    st.title("💰 Developer Salary Predictor")
    st.subheader("Welcome to the Developer Salary Prediction Dashboard")

    st.markdown("""
    This dashboard allows you to explore developer salary data from Stack Overflow surveys and predict salaries based on various factors.

    ### Dataset Overview
    The data is sourced from Stack Overflow Annual Developer Surveys from 2022 to 2024, providing insights into developer salaries worldwide.

    ### Features
    - **Data Exploration**: Visualize salary distributions and relationships with various factors
    - **Salary Prediction**: Predict your potential salary based on your profile
    - **Model Insights**: Understand what factors influence developer salaries the most

    ### Key Statistics
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(data):,}")

    with col2:
        st.metric("Average Salary", f"${data['Salary'].mean():,.2f}")

    with col3:
        st.metric("Countries", f"{data['Country'].nunique()}")

    with col4:
        st.metric("Avg Experience", f"{data['YearsExperience'].mean():.1f} years")

    st.subheader("Sample Data")
    st.dataframe(data.head(10))

def data_exploration_page():
    st.title("Data Exploration")

    tab1, tab2, tab3, tab4 = st.tabs(["Salary Distribution", "By Country", "By Experience", "Correlation Analysis"])

    with tab1:
        st.subheader("Salary Distribution")

        min_salary = int(data['Salary'].min())
        max_salary = int(data['Salary'].max())
        salary_range = st.slider("Filter Salary Range", min_salary, max_salary, (min_salary, max_salary))

        filtered_data = data[(data['Salary'] >= salary_range[0]) & (data['Salary'] <= salary_range[1])]

        fig = px.histogram(filtered_data, x="Salary", nbins=50, title="Distribution of Salaries")
        fig.update_layout(xaxis_title="Salary (USD)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(filtered_data, x="LogSalary", nbins=50, title="Distribution of Log Salaries")
        fig.update_layout(xaxis_title="Log Salary", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Salary by Country")

        min_country_count = st.slider("Minimum number of records per country", 10, 500, 100)
        country_counts = data['Country'].value_counts()
        top_countries = country_counts[country_counts > min_country_count].index.tolist()
        filtered_data = data[data['Country'].isin(top_countries)]

        country_data = filtered_data.groupby('Country')['Salary'].agg(['mean', 'median', 'count']).reset_index()
        country_data = country_data.sort_values('mean', ascending=False)

        viz_type = st.radio("Visualization Type", ["Bar Chart", "Map"], horizontal=True)

        if viz_type == "Bar Chart":
            fig = px.bar(country_data, x='Country', y='mean',
                         title="Average Salary by Country",
                         labels={'mean': 'Average Salary (USD)', 'Country': 'Country'},
                         hover_data=['median', 'count'])
            st.plotly_chart(fig, use_container_width=True)
        else:

            try:
                fig = px.choropleth(country_data,
                                    locations='Country',
                                    locationmode='country names',
                                    color='mean',
                                    hover_name='Country',
                                    hover_data=['median', 'count'],
                                    title='Average Developer Salary by Country',
                                    color_continuous_scale=px.colors.sequential.Plasma,
                                    labels={'mean': 'Average Salary (USD)'})
                fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating map visualization: {e}")
                st.info("Falling back to bar chart")
                fig = px.bar(country_data, x='Country', y='mean',
                             title="Average Salary by Country",
                             labels={'mean': 'Average Salary (USD)', 'Country': 'Country'},
                             hover_data=['median', 'count'])
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Salary by Experience")

        fig = px.scatter(data, x="YearsExperience", y="Salary",
                         title="Experience vs Salary",
                         labels={"YearsExperience": "Years of Experience", "Salary": "Salary (USD)"},
                         opacity=0.6,
                         trendline="lowess")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Correlation Analysis")

        numerical_cols = ['Salary', 'YearsExperience', 'LogSalary', 'ExpSquared']
        corr_data = data[numerical_cols].corr()

        fig = px.imshow(corr_data,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix of Numerical Variables")
        st.plotly_chart(fig, use_container_width=True)

        st.write("""
        **Observations:**
        - There's a strong positive correlation between Years of Experience and Salary
        - The logarithm of Salary (LogSalary) has an even stronger correlation with experience
        - ExpSquared (experience squared) shows the non-linear relationship between experience and salary
        """)

        st.subheader("Scatter Plot Matrix")
        try:
            fig = px.scatter_matrix(data,
                                    dimensions=['Salary', 'YearsExperience', 'LogSalary', 'ExpSquared'],
                                    title="Scatter Plot Matrix")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot matrix: {e}")
            st.info("This visualization requires more memory. Try filtering the data first.")

def salary_prediction_page():
    st.title("Salary Prediction")
    st.write("Use this tool to predict your potential salary based on your profile.")
    with st.container():
        st.subheader("Enter Your Information")

        col1, col2 = st.columns(2)

        with col1:

            countries = sorted(data['Country'].unique().tolist())
            top_countries = ['United States', 'Germany', 'United Kingdom', 'Canada', 'India', 'Australia']

            country_options = [c for c in top_countries if c in countries] + [c for c in countries if c not in top_countries]
            country = st.selectbox("Country", country_options, index=0 if 'United States' in country_options else 0)


            education_map = {
                "Bachelor's degree": "4-year degree (BS/BA)",
                "Master's degree": "Graduate degree (MS/MA/MEng)",
                "Doctoral degree": "PhD or equivalent",
                "Some college/university": "Partial university education",
                "High school": "Secondary education",
                "Professional degree": "JD, MD, etc.",
                "Associate degree": "2-year degree"
            }
            education_levels = sorted(data['EducationLevel'].unique().tolist())
            education_options = []
            for edu in education_levels:
                if edu in education_map:
                    education_options.append(f"{edu} ({education_map[edu]})")
                else:
                    education_options.append(edu)

            education_idx = st.selectbox("Education Level", range(len(education_options)),
                                        format_func=lambda x: education_options[x])
            education = education_levels[education_idx]


            experience = st.slider("Years of Professional Experience", 0, 30, 5,
                                help="Drag the slider to indicate your years of professional experience")

        with col2:

            if 'DeveloperType' in data.columns:
                dev_types = sorted(data['DeveloperType'].unique().tolist())
                dev_type_descriptions = {
                    "Backend Developer": "Server-side programming",
                    "Frontend Developer": "Client-side/UI programming",
                    "Full-stack Developer": "Both frontend and backend",
                    "Data Scientist": "Statistical analysis and ML",
                    "DevOps Engineer": "Infrastructure and deployment",
                    "Mobile Developer": "iOS, Android, or cross-platform",
                    "QA/Test Developer": "Quality assurance and testing"
                }

                dev_type_options = []
                for dt in dev_types:
                    if dt in dev_type_descriptions:
                        dev_type_options.append(f"{dt} ({dev_type_descriptions[dt]})")
                    else:
                        dev_type_options.append(dt)

                dev_type_idx = st.selectbox("Developer Type", range(len(dev_type_options)),
                                        format_func=lambda x: dev_type_options[x])
                dev_type = dev_types[dev_type_idx]
            else:
                dev_type = "Other"
            if 'CompanySize' in data.columns:
                company_sizes = sorted(data['CompanySize'].unique().tolist())
                company_size_map = {
                    "1-9 employees": "Micro (1-9)",
                    "10-99 employees": "Small (10-99)",
                    "100-999 employees": "Medium (100-999)",
                    "1,000-4,999 employees": "Large (1,000-4,999)",
                    "5,000-9,999 employees": "Very Large (5,000-9,999)",
                    "10,000+ employees": "Enterprise (10,000+)"
                }

                company_size_options = []
                for cs in company_sizes:
                    if cs in company_size_map:
                        company_size_options.append(f"{cs} ({company_size_map[cs]})")
                    else:
                        company_size_options.append(cs)

                company_size_idx = st.selectbox("Company Size", range(len(company_size_options)),
                                            format_func=lambda x: company_size_options[x])
                company_size = company_sizes[company_size_idx]
            else:
                company_size = "Unknown"


            if 'RemoteWork' in data.columns:
    
                remote_options = data['RemoteWork'].unique().tolist()

                remote_options = sorted(remote_options)

                remote_idx = st.selectbox("Remote Work Status", range(len(remote_options)),
                                        format_func=lambda x: remote_options[x])
                remote_work = remote_options[remote_idx]
            else:
                remote_work = "Unknown"

        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Predict My Salary", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    if predict_button:
        predicted_salary, _, _ = predict_salary(
            country, education, experience, dev_type, company_size, remote_work
        )

        st.subheader("Prediction Results")

        overall_p75 = data['Salary'].quantile(0.75)
        overall_p25 = data['Salary'].quantile(0.25)

        position_percent = min(100, max(0, (predicted_salary - overall_p25) / (overall_p75 - overall_p25) * 100))
        position_label = "Average"
        if position_percent < 33:
            position_label = "Lower Range"
        elif position_percent > 66:
            position_label = "Higher Range"

        marker_position = position_percent

        st.markdown(f'<div class="salary-value">${predicted_salary:,.0f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="salary-caption">Estimated annual salary in USD</div>', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="market-position">
            <span>Lower Range</span>
            <span>Average</span>
            <span>Higher Range</span>
        </div>
        <div class="position-indicator">
            <div class="position-marker" style="left: {marker_position}%;"></div>
        </div>
        <div style="text-align: center; margin-top: 5px; font-size: 0.9rem;">
            <strong>Market Position:</strong> {position_label}
        </div>
        ''', unsafe_allow_html=True)




def model_insights_page():
    st.title("Model Insights")
    st.header("Model Architecture")

    st.write("""
    The salary prediction model is a stacked ensemble of:
    - **XGBoost Regressor**: Handles non-linear relationships
    - **Random Forest Regressor**: Captures complex interactions

    These models were combined using a **Ridge regression** as the meta-learner.

    This ensemble approach provides more robust predictions than any single model.
    """)
    st.header("Performance Metrics")
    model_data = pd.DataFrame({
        'Model': ['XGBoost', 'Random Forest', 'Stacked Model'],
        'RMSE (log scale)': [0.4091, 0.4350, 0.4091],
        'R² Score': [0.6378, 0.5907, 0.6380],
        'RMSE (USD)': [38979.91, 40146.40, 38953.96]
    })
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1 = px.bar(
            model_data,
            x='Model',
            y='R² Score',
            title="R² Score (higher is better)",
            color='Model',
            color_discrete_sequence=['#0d47a1', '#1565c0', '#1976d2'],
            text_auto='.4f'
        )
        fig1.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            model_data,
            x='Model',
            y='RMSE (log scale)',
            title="RMSE (log scale) (lower is better)",
            color='Model',
            color_discrete_sequence=['#0d47a1', '#1565c0', '#1976d2'],
            text_auto='.4f'
        )
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.bar(
            model_data,
            x='Model',
            y='RMSE (USD)',
            title="RMSE (USD) (lower is better)",
            color='Model',
            color_discrete_sequence=['#0d47a1', '#1565c0', '#1976d2'],
            text_auto='$,.2f'
        )
        fig3.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.write("""
    **Observations:**
    - The **Stacked Model** achieves the best overall performance with the highest R² score (0.6380) and lowest RMSE in USD ($38,953.96)
    - **XGBoost** performs nearly as well as the Stacked Model
    - **Random Forest** has slightly lower performance metrics compared to the other models
    """)
    st.header("Model Limitations")

    st.info("""
    **Important limitations to consider:**

    1. **Data Representation**: The model is trained on Stack Overflow survey data, which may not represent all developers.

    2. **Self-Reported Data**: Salary information is self-reported and may contain inaccuracies.

    3. **Temporal Effects**: The model doesn't account for rapid changes in the job market or inflation.

    4. **Regional Variations**: While country is a feature, there can be significant variations within countries.

    5. **Industry-Specific Factors**: The model doesn't distinguish between industries, which can significantly impact salaries.
    """)

def main():
    page = navigation()

    if page == "Home":
        home_page()
    elif page == "Data Exploration":
        data_exploration_page()
    elif page == "Salary Prediction":
        salary_prediction_page()
    elif page == "Model Insights":
        model_insights_page()

if __name__ == "__main__":
    main()
