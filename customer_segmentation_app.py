import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation", layout="centered", initial_sidebar_state="auto")


dark_style = """
<style>
body {
    background-color: #0e1117;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
}
h1, h2, h3, h4 {
    color: #58a6ff;
}
</style>
"""
st.markdown(dark_style, unsafe_allow_html=True)


st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.markdown("Analyze mall customers by their **spending habits, income, and age** to discover valuable business segments.")


st.sidebar.header("ℹ️ What's this about?")
st.sidebar.write("""
Customer Segmentation helps businesses divide customers into groups based on their behavior. It can:
- 🎯 Improve targeted marketing
- 💰 Maximize profits
- 🤝 Boost customer satisfaction
""")


@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()


st.subheader("📄 Raw Dataset")
st.dataframe(df)


st.subheader("📊 Gender-based Spending")
fig = px.scatter(df, x='Annual Income (k$)', y='Spending Score (1-100)',
                 color='Gender', title="Spending Score vs Annual Income",
                 template="plotly_dark")
st.plotly_chart(fig)


df_model = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_model)


kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df['Cluster'] = clusters


st.subheader("📌 Clustered Segmentation (3D View)")
fig2 = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                     color='Cluster', title="Customer Clusters in 3D", template="plotly_dark")
st.plotly_chart(fig2)


st.subheader("📈 Number of Customers per Cluster")
st.bar_chart(df['Cluster'].value_counts().sort_index())


st.subheader("🔎 Explore by Cluster")
cluster_choice = st.selectbox("Select a cluster to explore:", sorted(df['Cluster'].unique()))
filtered_df = df[df['Cluster'] == cluster_choice]
st.write(f"Showing customers from **Cluster {cluster_choice}**")
st.dataframe(filtered_df)


st.subheader("📌 What Each Cluster Means (Business Insights)")
with st.expander("💡 Show Cluster Descriptions"):
    st.markdown("""
- 🟢 **Cluster 0:** High Income, Low Spending – Upsell through premium services.
- 🔵 **Cluster 1:** Young & High Spending – Engage for luxury/premium campaigns.
- 🟡 **Cluster 2:** Budget-Conscious – Offer loyalty programs & discounts.
- 🟣 **Cluster 3:** VIP – High income & High spenders. Retain with exclusive offers.
- 🟠 **Cluster 4:** Moderate customers – Educate, cross-sell products.
""")
    
st.subheader("💾 Download Segmented Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download CSV with Clusters",
    data=csv,
    file_name='clustered_customers.csv',
    mime='text/csv',
)

st.markdown("---")
st.markdown("🚀 Built with ❤️ for Internship Task 4 | By Muneeb Adil😄")
