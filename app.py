import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kneed import KneeLocator

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df_books = load_data('data/clean/df_books_clean.csv')
df_books_cluster = load_data('data/clean/df_books_cluster.csv')
df_books_scaled = load_data('data/clean/df_books_clean.csv')

st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] {
            justify-content: left;
        }
        .stMarkdown, .stPlotlyChart, .stDataFrame {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

data_features = df_books.drop(columns=["title", "author"]).values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_features)

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=1000)
df_books["cluster"] = kmeans.fit_predict(data_scaled)

# Metrics: Silhouette Score
silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)

# Optimal number of clusters: Elbow method
inertias = []
range_of_clusters = range(1, 21)

for k in range_of_clusters:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data_scaled)
    inertias.append(model.inertia_)

# Finding the optimal number of clusters using the KneeLocator
kn = KneeLocator(range_of_clusters, inertias, curve='convex', direction='decreasing')
optimal_clusters = kn.knee

# Recommender Function
def recommend_similar_book(book_name, df):
    cluster_row = df.loc[df['title'] == book_name]
    return df[df["cluster"] == cluster_row.cluster.values[0]].sample(5)  # Return 5 random recommendations


# -----------------------------------------------

st.title("BookReco")
# st.subheader("The Book Recommendation System")

tab1, tab2, tab3 = st.tabs(["Recommend me a book", "How it works", "All our books"])

with tab1:
    st.subheader("Recommend me a book")
    # st.write("This dashboard visualizes the Iris dataset with interactive charts and tables.")
    # st.write("### 📊 Data Overview")
    # st.write(df.describe())
    # st.metric("Total Samples", df.shape[0])
    # st.metric("Total Features", df.shape[1] - 1)
    # Get a recommendation for a random book

    title_list = list(df_books["title"])

    book_name = st.selectbox(
    "Select a book from the list to get similar titles",
    title_list,
    index=None,
    placeholder="Click here to see the books...",
    )

    # st.write(book_name)

    if book_name != None:

        # book_name = "Peter Pan"	  # Choose an index
        df_recommend = recommend_similar_book(book_name, df_books_cluster)

        st.dataframe(df_recommend[["title", "author", "first_publish_year", "rating"]])

with tab2:
    st.subheader("Process")

    # chart_type = st.radio("Select Chart Type", ["Scatter Plot", "Box Plot"])


    fig = px.scatter(df_books_cluster, x="first_publish_year", y="rating", title="Rating level through the years")
    st.plotly_chart(fig, use_container_width=True)
    

    fig = px.scatter(df_books_cluster, x="first_publish_year", y="rating", color="cluster", title="Rating level through the years (in clusters)")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Metrics")

    st.subheader(f"Silhouette Score: {silhouette_avg:.3f}")

    st.subheader("Optimal number of clusters: Elbow method")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range_of_clusters), y=inertias, mode='lines+markers', name='Inertia'))
    fig.update_layout(title='Elbow Method For Optimal k',
                    xaxis_title='Number of clusters, k',
                    yaxis_title='Inertia',
                    xaxis=dict(tickmode='array', tickvals=list(range_of_clusters)))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Knee method optimal clusters: {optimal_clusters}")

with tab3:
    st.subheader(f"Our {df_books.shape[0]} books")
    st.dataframe(df_books, use_container_width=True)
