import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from groq import Groq


st.set_page_config(
    page_title="ExperimentOS",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }
    .feature-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.20);
        background: rgba(128,128,128,0.06);
        min-height: 130px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "plan" not in st.session_state:
    st.session_state.plan = None

if "interpretation" not in st.session_state:
    st.session_state.interpretation = None


def get_groq_client():
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        return None

    api_key = str(api_key).strip()

    if not api_key:
        return None

    return Groq(api_key=api_key)


client = get_groq_client()


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def generate_experiment_plan(question, mode, columns):
    if client is None:
        return {
            "error": (
                "Groq API key is not configured. "
                "Add GROQ_API_KEY in Streamlit Cloud Secrets."
            )
        }

    prompt = f"""
You are the AI planning engine for ExperimentOS.

User type:
{mode}

User question:
{question}

Available dataset columns:
{columns}

Create a practical experiment/research plan.

IMPORTANT:
- Only use column names that actually exist in the dataset.
- Do not invent columns.
- If the question cannot be properly answered from the available columns,
  clearly mention that.
- Keep the plan understandable for both students and business users.

Return ONLY valid JSON.

Required structure:

{{
    "research_question": "...",
    "objective": "...",
    "hypothesis": "...",
    "null_hypothesis": "...",
    "independent_variables": [],
    "dependent_variables": [],
    "possible_confounders": [],
    "recommended_analysis": [],
    "data_requirements": [],
    "limitations": []
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful statistical experiment planner. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )

        result = extract_json(
            response.choices[0].message.content
        )

        if result is None:
            return {"error": "AI returned an invalid JSON response."}

        return result

    except Exception as e:
        return {"error": f"Groq request failed: {str(e)}"}


def interpret_results(question, results):
    if client is None:
        return (
            "AI interpretation unavailable because GROQ_API_KEY "
            "is not configured."
        )

    prompt = f"""
You are the interpretation engine of ExperimentOS.

Research/business question:
{question}

Actual Python-computed results:
{json.dumps(results, indent=2)}

Explain these results to a non-technical user.

Rules:
1. Clearly state the main finding.
2. Explain statistical significance if a p-value exists.
3. Explain practical meaning.
4. Never claim causation from correlation alone.
5. Mention limitations.
6. Do not invent numbers.
7. If evidence is weak, say so.
8. Keep the explanation concise and professional.

Use these headings:

### Main Finding
### What the Statistics Mean
### Practical Meaning
### Limitations
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an accurate statistical results interpreter."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI interpretation failed: {str(e)}"


def profile_dataset(df):
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def run_correlation(df, x_col, y_col):
    temp = df[[x_col, y_col]].copy()

    temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 3:
        return None

    if temp[x_col].nunique() < 2 or temp[y_col].nunique() < 2:
        return None

    correlation, p_value = stats.pearsonr(
        temp[x_col],
        temp[y_col]
    )

    return {
        "sample_size": len(temp),
        "x_variable": x_col,
        "y_variable": y_col,
        "pearson_correlation": float(correlation),
        "p_value": float(p_value),
    }


def run_regression(df, x_col, y_col):
    temp = df[[x_col, y_col]].copy()

    temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    temp = temp.dropna()

    if len(temp) < 3 or temp[x_col].nunique() < 2:
        return None

    X = temp[[x_col]]
    y = temp[y_col]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    return {
        "sample_size": len(temp),
        "x_variable": x_col,
        "y_variable": y_col,
        "coefficient": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r_squared": float(r2_score(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
    }


def run_ttest(df, group_col, value_col):
    temp = df[[group_col, value_col]].copy()
    temp[value_col] = pd.to_numeric(
        temp[value_col],
        errors="coerce"
    )
    temp = temp.dropna()

    groups = temp[group_col].dropna().unique()

    if len(groups) != 2:
        return None

    group_a = temp[temp[group_col] == groups[0]][value_col]
    group_b = temp[temp[group_col] == groups[1]][value_col]

    if len(group_a) < 2 or len(group_b) < 2:
        return None

    statistic, p_value = stats.ttest_ind(
        group_a,
        group_b,
        equal_var=False
    )

    return {
        "group_variable": group_col,
        "value_variable": value_col,
        "group_a": str(groups[0]),
        "group_b": str(groups[1]),
        "group_a_size": int(len(group_a)),
        "group_b_size": int(len(group_b)),
        "group_a_mean": float(group_a.mean()),
        "group_b_mean": float(group_b.mean()),
        "t_statistic": float(statistic),
        "p_value": float(p_value),
    }


st.markdown(
    '<div class="main-title">🧪 ExperimentOS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Experimentation & Decision Intelligence Platform'
    '</div>',
    unsafe_allow_html=True
)


st.sidebar.title("ExperimentOS")

mode = st.sidebar.radio(
    "Who are you?",
    [
        "🎓 Student / Researcher",
        "💼 Business / Company",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### How it works

**1.** Ask a question  
**2.** Upload your CSV  
**3.** Generate an AI experiment plan  
**4.** Check data quality  
**5.** Run statistical analysis  
**6.** Visualize results  
**7.** Get an AI explanation
"""
)

st.sidebar.markdown("---")

if client:
    st.sidebar.success("Groq AI: Connected")
else:
    st.sidebar.warning("Groq AI: Not configured")


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
        <h3>🧠 AI Planning</h3>
        Turn natural-language questions into structured experiments.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
        <h3>📊 Real Statistics</h3>
        Python performs the actual statistical calculations.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
        <h3>💡 Decision Support</h3>
        Understand what your data actually says.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="section-title">1. Define Your Question</div>',
    unsafe_allow_html=True
)

placeholder = (
    "Example: Does study time affect exam performance?"
    if mode == "🎓 Student / Researcher"
    else "Example: Does a discount increase customer purchases?"
)

question = st.text_area(
    "What do you want to investigate?",
    placeholder=placeholder,
    height=120,
)


st.markdown(
    '<div class="section-title">2. Upload Your Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
    help="CSV files are currently supported.",
)


if uploaded_file is None:
    st.info("Upload a CSV file to start your experiment.")

    st.markdown("---")
    st.markdown("### Example questions")

    if mode == "🎓 Student / Researcher":
        st.markdown(
            """
            - Does study time affect exam score?
            - Does attendance relate to academic performance?
            - Do two student groups have different average scores?
            """
        )
    else:
        st.markdown(
            """
            - Does discount percentage affect sales?
            - Does delivery time affect customer satisfaction?
            - Which factors are associated with revenue?
            """
        )

else:
    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV appears to be empty.")
            st.stop()

        profile = profile_dataset(df)

        st.markdown("---")

        st.markdown(
            '<div class="section-title">3. Dataset Overview</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", f"{profile['rows']:,}")
        c2.metric("Columns", profile["columns"])
        c3.metric("Missing Values", f"{profile['missing_values']:,}")
        c4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")

        with st.expander("Preview Dataset", expanded=True):
            st.dataframe(
                df.head(15),
                use_container_width=True
            )

        st.markdown(
            '<div class="section-title">4. Data Quality</div>',
            unsafe_allow_html=True
        )

        quality_score = 100

        if profile["missing_values"] > 0:
            quality_score -= 15

        if profile["duplicate_rows"] > 0:
            quality_score -= 10

        if profile["rows"] < 30:
            quality_score -= 20

        quality_score = max(0, quality_score)

        q1, q2, q3 = st.columns(3)

        q1.metric("Data Quality Score", f"{quality_score}/100")
        q2.metric("Numeric Variables", len(profile["numeric_columns"]))
        q3.metric(
            "Categorical Variables",
            len(profile["categorical_columns"])
        )

        if profile["missing_values"] > 0:
            st.warning(
                f"This dataset contains "
                f"{profile['missing_values']:,} missing values."
            )
        else:
            st.success("No missing values detected.")

        if profile["duplicate_rows"] > 0:
            st.warning(
                f"{profile['duplicate_rows']:,} duplicate rows detected."
            )

        if profile["rows"] < 30:
            st.warning(
                "The dataset contains fewer than 30 rows. "
                "Statistical conclusions may be unstable."
            )

        st.markdown(
            '<div class="section-title">5. AI Experiment Planner</div>',
            unsafe_allow_html=True
        )

        if not question.strip():
            st.info("Enter a research or business question above.")

        if st.button(
            "🧠 Generate Experiment Plan",
            type="primary",
            use_container_width=True,
        ):
            if not question.strip():
                st.warning("Please enter your question first.")
            elif client is None:
                st.error(
                    "Groq API is not configured. "
                    "Add GROQ_API_KEY in Streamlit Cloud Secrets."
                )
            else:
                with st.spinner(
                    "ExperimentOS is designing your experiment..."
                ):
                    st.session_state.plan = generate_experiment_plan(
                        question,
                        mode,
                        df.columns.tolist(),
                    )

                st.session_state.interpretation = None

        if st.session_state.plan:
            plan = st.session_state.plan

            if "error" in plan:
                st.error(plan["error"])
            else:
                st.markdown("---")
                st.subheader("🧠 Your Experiment Plan")

                st.write("**Research Question**")
                st.info(
                    plan.get("research_question", "Not available")
                )

                st.write("**Objective**")
                st.write(
                    plan.get("objective", "Not available")
                )

                st.write("**Hypothesis**")
                st.write(
                    plan.get("hypothesis", "Not available")
                )

                st.write("**Null Hypothesis**")
                st.write(
                    plan.get("null_hypothesis", "Not available")
                )

                p1, p2 = st.columns(2)

                with p1:
                    st.write("**Independent Variables**")
                    variables = plan.get(
                        "independent_variables", []
                    )
                    if variables:
                        for item in variables:
                            st.write(f"- {item}")
                    else:
                        st.write("None identified.")

                with p2:
                    st.write("**Dependent Variables**")
                    variables = plan.get(
                        "dependent_variables", []
                    )
                    if variables:
                        for item in variables:
                            st.write(f"- {item}")
                    else:
                        st.write("None identified.")

                st.write("**Recommended Analysis**")
                for item in plan.get(
                    "recommended_analysis", []
                ):
                    st.write(f"- {item}")

                st.write("**Possible Confounders**")
                for item in plan.get(
                    "possible_confounders", []
                ):
                    st.write(f"- {item}")

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '6. Statistical Analysis'
                    '</div>',
                    unsafe_allow_html=True
                )

                numeric_columns = profile["numeric_columns"]
                categorical_columns = profile["categorical_columns"]

                analysis_tabs = st.tabs(
                    [
                        "📈 Correlation & Regression",
                        "🧪 Two-Group Test",
                    ]
                )

                with analysis_tabs[0]:
                    if len(numeric_columns) < 2:
                        st.info(
                            "At least two numeric columns are required."
                        )
                    else:
                        a1, a2 = st.columns(2)

                        with a1:
                            x_col = st.selectbox(
                                "Independent variable (X)",
                                numeric_columns,
                                key="correlation_x",
                            )

                        possible_y = [
                            c for c in numeric_columns
                            if c != x_col
                        ]

                        with a2:
                            y_col = st.selectbox(
                                "Dependent variable (Y)",
                                possible_y,
                                key="correlation_y",
                            )

                        correlation_result = run_correlation(
                            df,
                            x_col,
                            y_col
                        )

                        regression_result = run_regression(
                            df,
                            x_col,
                            y_col
                        )

                        if correlation_result is None:
                            st.error(
                                "Unable to calculate correlation "
                                "for these variables."
                            )
                        else:
                            st.subheader("Pearson Correlation")

                            m1, m2, m3 = st.columns(3)

                            m1.metric(
                                "Correlation",
                                f"{correlation_result['pearson_correlation']:.4f}"
                            )

                            m2.metric(
                                "P-Value",
                                f"{correlation_result['p_value']:.6f}"
                            )

                            m3.metric(
                                "Sample Size",
                                correlation_result["sample_size"]
                            )

                            if correlation_result["p_value"] < 0.05:
                                st.success(
                                    "The correlation is statistically "
                                    "significant at the 0.05 level."
                                )
                            else:
                                st.info(
                                    "The correlation is not statistically "
                                    "significant at the 0.05 level."
                                )

                            chart_df = df[[x_col, y_col]].copy()

                            chart_df[x_col] = pd.to_numeric(
                                chart_df[x_col],
                                errors="coerce"
                            )

                            chart_df[y_col] = pd.to_numeric(
                                chart_df[y_col],
                                errors="coerce"
                            )

                            chart_df = chart_df.dropna()

                            fig = px.scatter(
                                chart_df,
                                x=x_col,
                                y=y_col,
                                trendline="ols",
                                title=f"{x_col} vs {y_col}",
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            if regression_result:
                                st.subheader("Linear Regression")

                                r1, r2, r3 = st.columns(3)

                                r1.metric(
                                    "R²",
                                    f"{regression_result['r_squared']:.4f}"
                                )

                                r2.metric(
                                    "RMSE",
                                    f"{regression_result['rmse']:.4f}"
                                )

                                r3.metric(
                                    "Coefficient",
                                    f"{regression_result['coefficient']:.4f}"
                                )

                                st.caption(
                                    "R² describes the proportion of variation "
                                    "in the selected outcome explained by "
                                    "this simple linear model."
                                )

                                actual_results = {
                                    "correlation": correlation_result,
                                    "regression": regression_result,
                                }

                                st.subheader("🤖 AI Interpretation")

                                if st.button(
                                    "Generate Evidence Explanation",
                                    key="interpret_button",
                                    use_container_width=True,
                                ):
                                    with st.spinner(
                                        "AI is interpreting the evidence..."
                                    ):
                                        st.session_state.interpretation = (
                                            interpret_results(
                                                question,
                                                actual_results
                                            )
                                        )

                                if st.session_state.interpretation:
                                    st.markdown(
                                        st.session_state.interpretation
                                    )

                with analysis_tabs[1]:
                    if not categorical_columns:
                        st.info(
                            "No categorical columns were detected."
                        )
                    elif not numeric_columns:
                        st.info(
                            "No numeric columns were detected."
                        )
                    else:
                        group_col = st.selectbox(
                            "Grouping variable",
                            categorical_columns,
                            key="ttest_group",
                        )

                        value_col = st.selectbox(
                            "Numeric outcome",
                            numeric_columns,
                            key="ttest_value",
                        )

                        result = run_ttest(
                            df,
                            group_col,
                            value_col
                        )

                        if result is None:
                            st.info(
                                "The selected grouping variable must contain "
                                "exactly two groups with enough observations."
                            )
                        else:
                            t1, t2, t3 = st.columns(3)

                            t1.metric(
                                "Group A Mean",
                                f"{result['group_a_mean']:.3f}"
                            )

                            t2.metric(
                                "Group B Mean",
                                f"{result['group_b_mean']:.3f}"
                            )

                            t3.metric(
                                "P-Value",
                                f"{result['p_value']:.6f}"
                            )

                            if result["p_value"] < 0.05:
                                st.success(
                                    "The difference between the two groups "
                                    "is statistically significant at the "
                                    "0.05 level."
                                )
                            else:
                                st.info(
                                    "The difference between the two groups "
                                    "is not statistically significant at "
                                    "the 0.05 level."
                                )

                            box_df = df[[group_col, value_col]].copy()

                            box_df[value_col] = pd.to_numeric(
                                box_df[value_col],
                                errors="coerce"
                            )

                            box_df = box_df.dropna()

                            fig = px.box(
                                box_df,
                                x=group_col,
                                y=value_col,
                                title=f"{value_col} by {group_col}",
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                st.markdown("---")
                st.subheader("📥 Export Experiment Plan")

                report = {
                    "platform": "ExperimentOS",
                    "mode": mode,
                    "question": question,
                    "dataset_rows": profile["rows"],
                    "dataset_columns": profile["columns"],
                    "data_quality": {
                        "missing_values": profile["missing_values"],
                        "duplicate_rows": profile["duplicate_rows"],
                        "quality_score": quality_score,
                    },
                    "experiment_plan": plan,
                }

                st.download_button(
                    label="Download Experiment JSON",
                    data=json.dumps(
                        report,
                        indent=2,
                        default=str
                    ),
                    file_name="experimentOS_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

    except Exception as e:
        st.error("The dataset could not be processed.")
        st.exception(e)


st.markdown("---")

st.caption(
    "ExperimentOS is a decision-support platform. "
    "Statistical results should be interpreted in context. "
    "Correlation does not by itself establish causation."
)
