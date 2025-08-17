import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ------------------- Sidebar Navigation -------------------
option = st.sidebar.radio(
    label="Go to",
    options=["Home Page", "EDA & Plots", "Preprocessed Data"]
)

# ------------------- Home Page -------------------
def home_page():
    st.title("Real Estate Data Analysis - Aqarmap Project")
    st.subheader("Project Introduction")
    intro_text = """
    This project analyzes real estate listings scraped from **Aqarmap**.

    **Dataset Overview:**
    - Contains property details such as: price, price per meter, location, area, bedrooms, bathrooms, and more.
    - Data includes both **numerical** and **categorical** features.

    **Key Insights from EDA:**
    - **Average property price:** 7,432,927.8 EGP.
    - Most listings are located in **New Cairo**.
    - Typical apartment area around **149 sqm**.
    - **3rd floor** is the most frequently listed floor.
    - **Ain Sokhna** has the highest price per meter.
    - Among the top 10 most common property sizes, **160 sqm** is the most listed.

    Suitable for **EDA**, **visualization**, and **predictive modeling** after preprocessing and scaling.
    """
    st.write(intro_text)

    st.subheader("Scraped Real Estate Dataset")
    df = pd.read_csv('aqarmap_all_apartments_merged.csv')  
    st.dataframe(df)

# ------------------- EDA & Plots -------------------
def eda_page():
    st.title("Exploratory Data Analysis")
    df = pd.read_csv('aqar_deployment.csv')

    # ------------------- Convert columns safely -------------------
    df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(",", ""), errors='coerce')
    df['Area in m²'] = pd.to_numeric(df['Area in m²'].astype(str).str.replace(" m²",""), errors='coerce')
    df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')

    tabs = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

    # --- Univariate Analysis ---
    with tabs[0]:
        st.subheader("Numeric Columns")
        num_columns = df.select_dtypes(include="number").columns
        for col in num_columns:
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            sns.histplot(df[col], ax=axes[0], kde=True)
            axes[0].set_title(f'Distribution of {col}')
            sns.boxplot(y=df[col], ax=axes[1])
            axes[1].set_title(f'Boxplot of {col}')
            st.pyplot(fig)

        st.subheader("Categorical Columns")
        cat_columns = df.select_dtypes(include=["object", "string"]).columns
        for col in cat_columns:
            unique_vals = df[col].nunique()
            st.markdown(f"**{col}**")
            if unique_vals < 7:
                fig = px.pie(df, names=col, title=f'Pie Chart of {col}')
                st.plotly_chart(fig)
            elif unique_vals > 10:
                fig = px.histogram(df, x=col, title=f'Count Plot of {col}')
                st.plotly_chart(fig)
            else:
                fig = px.bar(df[col].value_counts().reset_index(),
                             x="index", y=col,
                             title=f'Bar Chart of {col}')
                st.plotly_chart(fig)

    # --- Bivariate Analysis ---
    with tabs[1]:
        st.subheader("Bivariate Analysis")

        # 1) Price Distribution by View
        st.markdown("**1) Price Distribution by View**")
        fig1 = px.box(df, x='View', y='Price', title='Price Distribution by View')
        st.plotly_chart(fig1)
        st.markdown("Most of the data is between 10-30M, but the most expensive is an apartment with main street view.")

        # 2) Average Price per Square Meter by Governorate
        st.markdown("**2) Average Price per Square Meter by Governorate**")
        avg_price_per_meter = df.groupby('Governorate', as_index=False).agg(Avg_Price_per_Meter=('Price per meter', 'mean'))
        fig2 = px.bar(avg_price_per_meter, x='Governorate', y='Avg_Price_per_Meter', title='Average price per square meter for each governorate')
        st.plotly_chart(fig2)

        # 3) Average Property Price by Payment Method
        st.markdown("**3) Average Property Price by Payment Method**")
        avg_price_payment = df.groupby('Payment Method', as_index=False)['Price'].mean()
        fig3, ax3 = plt.subplots(figsize=(8,6))
        sns.barplot(data=avg_price_payment, x='Payment Method', y='Price', ax=ax3)
        ax3.set_title('Average Property Price by Payment Method')
        ax3.set_xlabel('Payment Method')
        ax3.set_ylabel('Average Price')
        st.pyplot(fig3)
        st.markdown("Properties paid in cash are lower priced than those purchased via installment or other methods.")

        # 4) Number of Ads by Governorate
        st.markdown("**4) Number of Ads by Governorate**")
        gov_counts = df['Governorate'].value_counts().reset_index()
        gov_counts.columns = ['Governorate', 'Count']
        fig4 = px.bar(gov_counts, x='Governorate', y='Count', text='Count', title='Number of Ads by Governorate')
        st.plotly_chart(fig4)

        # 5) Price across Year Built
        st.markdown("**5) Price across Year Built**")
        fig5 = px.scatter(df, y='Price', x='Year Built', title='Price across Year Built')
        st.plotly_chart(fig5)

        # 6) Most Advertised Floors
        st.markdown("**6) Most Advertised Floors**")
        fig6, ax6 = plt.subplots(figsize=(8,6))
        sns.countplot(data=df, x='Floor', ax=ax6)
        ax6.set_title('Most Advertised Floors')
        ax6.set_xlabel('Floor')
        ax6.set_ylabel('Number of Ads')
        st.pyplot(fig6)

        # 7) Relationship between Price and Floor
        st.markdown("**7) Relationship between Price and Floor**")
        fig7 = px.scatter(df, x='Floor', y='Price', title='Relationship between Price and Floor')
        st.plotly_chart(fig7)

        # 8) Top 10 Most Common Areas
        st.markdown("**8) Top 10 Most Common Areas**")
        area_counts = df['Area in m²'].value_counts().reset_index()
        area_counts.columns = ['Area in m²', 'Count']
        fig8, ax8 = plt.subplots(figsize=(10,6))
        sns.barplot(data=area_counts.head(10), x='Area in m²', y='Count', ax=ax8)
        ax8.set_title('Top 10 Most Common Areas')
        ax8.set_xlabel('Area in m²')
        ax8.set_ylabel('Count')
        st.pyplot(fig8)

        # 9) Price Distribution in North Coast, Alexandria, and Marsa Matruh
        st.markdown("**9) Property Price Distribution in Selected Governorates**")
        selected_govs = ['North Coast', 'Alexandria', 'Marsa Matruh']
        df_selected = df[df['Governorate'].isin(selected_govs)]
        fig9 = px.box(df_selected, x='Governorate', y='Price', title='Property Price Distribution in North Coast, Alexandria, and Marsa Matruh')
        st.plotly_chart(fig9)

        # 10) Price vs Number of Bedrooms
        st.markdown("**10) Price vs Number of Bedrooms**")
        fig10, ax10 = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=df, x='Bedrooms', y='Price', ax=ax10)
        ax10.set_title('Price vs Number of Bedrooms')
        st.pyplot(fig10)

    # --- Multivariate Analysis ---
    with tabs[2]:
        st.subheader("Multivariate Analysis - Numeric Columns")
        fig = sns.pairplot(df.select_dtypes("number"))
        st.pyplot(fig.fig)

# ------------------- Preprocessed Data -------------------
def preprocessed_page():
    st.title("Preprocessed Data")
    df_final = pd.read_csv('Full_train_set.csv')
    st.dataframe(df_final)

# ------------------- Navigation -------------------
page_dict = {
    "Home Page": home_page,
    "EDA & Plots": eda_page,
    "Preprocessed Data": preprocessed_page
}

page_dict[option]()



