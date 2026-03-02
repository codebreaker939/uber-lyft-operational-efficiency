import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(layout="wide", page_title="Uber vs Lyft Playground")

st.markdown("""
<style>
.block-container { padding: 1rem 1rem 0 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──
@st.cache_data
def load_data():
    ds = pd.read_csv("rideshare_kaggle.csv")
    ds.dropna(subset=["price"], inplace=True)
    ds["price_per_mile"] = ds["price"] / ds["distance"]
    return ds

ds = load_data()

# ── Session state init ──
for i in range(1, 11):
    if f"show_{i}" not in st.session_state:
        st.session_state[f"show_{i}"] = False
    if f"run_{i}" not in st.session_state:
        st.session_state[f"run_{i}"] = False

st.title("Uber vs Lyft Boston — Analysis Playground")

left, right = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════
# LEFT SIDE — All Code Cells
# ══════════════════════════════════════════
steps = [
    {"id": 1, "title": "Step 1 — Data Overview",
     "code": "ds.head()\nds.shape\nds.describe()"},
    {"id": 2, "title": "Step 2 — Average Price: Uber vs Lyft",
     "code": 'avg = ds.groupby("cab_type")["price"].mean()\navg.plot(kind="bar")'},
    {"id": 3, "title": "Step 3 — Price Per Mile",
     "code": 'ds["price_per_mile"] = ds["price"] / ds["distance"]\nds.groupby("cab_type")["price_per_mile"].mean()'},
    {"id": 4, "title": "Step 4 — Price by Hour",
     "code": 'ds.groupby(["hour","cab_type"])["price"].mean().unstack().plot()'},
    {"id": 5, "title": "Step 5 — Surge Multiplier",
     "code": 'ds.groupby("cab_type")["surge_multiplier"].mean()'},
    {"id": 6, "title": "Step 6 — Peak Hour Demand",
     "code": 'ds.groupby(["hour","cab_type"]).size().unstack().plot()'},
    {"id": 7, "title": "Step 7 — K-Means Clustering",
     "code": 'kmeans = KMeans(n_clusters=4, random_state=42)\ncd["cluster"] = kmeans.fit_predict(X_scaled)\ncd.groupby("cluster").mean()'},
    {"id": 8, "title": "Step 8 — Logistic Regression",
     "code": 'model = LogisticRegression()\nmodel.fit(X_train, y_train)\nprint(accuracy_score(y_test, y_pred))\nprint(classification_report(y_test, y_pred))'},
    {"id": 9, "title": "Step 9 — Cluster Visualization",
     "code": 'sns.scatterplot(data=cd, x="hour", y="price",\n    hue="Cluster Type", palette="Set1")\nplt.title("Ride Clusters — Hour vs Price")'},
    {"id": 10, "title": "Step 10 — 🎯 Price Predictor",
     "code": 'model.predict([[hour, distance, surge]])\nmodel.predict_proba([[hour, distance, surge]])'},
]

with left:
    st.markdown("##Code Cells")
    for step in steps:
        i = step["id"]
        st.markdown(f"**{step['title']}**")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("📄 Code", key=f"code_btn_{i}"):
                st.session_state[f"show_{i}"] = not st.session_state[f"show_{i}"]
        with c2:
            if st.button("▶ Run", key=f"run_btn_{i}"):
                for j in range(1, 9):
                    st.session_state[f"run_{j}"] = False
                st.session_state[f"run_{i}"] = True
        if st.session_state[f"show_{i}"]:
            st.code(step["code"], language="python")
        st.markdown("---")

# ══════════════════════════════════════════
# RIGHT SIDE — All Outputs
# ══════════════════════════════════════════
with right:
    st.markdown("## Output")

    # ── Step 1 ──
    if st.session_state["run_1"]:
        st.markdown("**📌 Step 1 — Data Overview**")
        st.write("**Shape:**", ds.shape)
        st.dataframe(ds.head())
        st.dataframe(ds.describe())
        st.markdown("---")

    # ── Step 2 ──
    if st.session_state["run_2"]:
        st.markdown("**📌 Step 2 — Average Price**")
        avg = ds.groupby("cab_type")["price"].mean()
        st.dataframe(avg)
        fig, ax = plt.subplots()
        avg.plot(kind="bar", color=["black", "pink"], edgecolor="white", ax=ax)
        ax.set_title("Average Price: Uber vs Lyft")
        ax.set_ylabel("Price ($)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        st.info("💡 Lyft ($17.35) is on average $1.55 more expensive than Uber ($15.80)")
        st.markdown("---")

    # ── Step 3 ──
    if st.session_state["run_3"]:
        st.markdown("**📌 Step 3 — Price Per Mile**")
        ppm = ds.groupby("cab_type")["price_per_mile"].mean()
        st.dataframe(ppm)
        fig, ax = plt.subplots()
        ppm.plot(kind="bar", color=["black", "pink"], edgecolor="white", ax=ax)
        ax.set_title("Price Per Mile: Uber vs Lyft")
        ax.set_ylabel("$ per Mile")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        st.info("💡 Price per mile is nearly identical — price gap comes from base fare, not distance")
        st.markdown("---")

    # ── Step 4 ──
    if st.session_state["run_4"]:
        st.markdown("**📌 Step 4 — Price by Hour**")
        hourly = ds.groupby(["hour", "cab_type"])["price"].mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 4))
        hourly.plot(ax=ax, color=["black", "pink"])
        ax.set_title("Average Price by Hour")
        ax.set_ylabel("Price ($)")
        ax.set_xlabel("Hour of Day")
        st.pyplot(fig)
        st.info("💡 Uber is cheapest at 12 PM ($15.69). Lyft is cheapest at 6 PM ($17.24)")
        st.markdown("---")

    # ── Step 5 ──
    if st.session_state["run_5"]:
        st.markdown("**📌 Step 5 — Surge Multiplier**")
        surge = ds.groupby("cab_type")["surge_multiplier"].mean()
        st.dataframe(surge)
        fig, ax = plt.subplots()
        surge.plot(kind="bar", color=["black", "pink"], edgecolor="white", ax=ax)
        ax.set_title("Average Surge Multiplier: Uber vs Lyft")
        ax.set_ylabel("Surge Multiplier")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        st.info("💡 Uber surge = 1.00 (never surges). Lyft = 1.03 (slightly surges)")
        st.markdown("---")

    # ── Step 6 ──
    if st.session_state["run_6"]:
        st.markdown("**📌 Step 6 — Peak Hour Demand**")
        peak = ds.groupby(["hour", "cab_type"]).size().unstack()
        fig, ax = plt.subplots(figsize=(8, 4))
        peak.plot(ax=ax, color=["black", "pink"])
        ax.set_title("Ride Demand by Hour of Day")
        ax.set_ylabel("Number of Rides")
        ax.set_xlabel("Hour of Day")
        st.pyplot(fig)
        st.info("💡 Uber dominates ride volume at every hour across Boston")
        st.markdown("---")

    # ── Step 7 ──
    if st.session_state["run_7"]:
        st.markdown("**📌 Step 7 — K-Means Clustering**")
        cd = ds[["hour", "price", "distance"]].dropna().copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cd)
        km = KMeans(n_clusters=4, random_state=42)
        cd["cluster"] = km.fit_predict(X_scaled)

        cluster_labels = {
            0: "🌆 Evening Short Rides",
            1: "🌅 Early Morning Short",
            2: "💼 Afternoon Long Expensive",
            3: "🏙️ Morning Long Cheap"
        }
        cd["Cluster Type"] = cd["cluster"].map(cluster_labels)

        summary = cd.groupby("Cluster Type")[["hour", "price", "distance"]].mean().round(2)
        st.dataframe(summary)
        st.info("💡 Cluster 2 (Afternoon Long Rides) is the most profitable at avg $31.15")
        st.markdown("---")

    # ── Step 8 ──
    if st.session_state["run_8"]:
        st.markdown("**📌 Step 8 — Logistic Regression**")
        ds2 = ds.copy()
        ds2["expensive"] = (ds2["price"] > ds2["price"].mean()).astype(int)
        X = ds2[["hour", "distance", "surge_multiplier"]].dropna()
        y = ds2.loc[X.index, "expensive"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {acc * 100:.2f}%")
        st.text(classification_report(y_test, y_pred))
        st.info("💡 Model predicts whether a ride will be above or below average price")
        st.markdown("---")

    # ── Step 9 — Cluster Visualization ──
    if st.session_state["run_9"]:
        st.markdown("**📌 Step 9 — Cluster Visualization**")
        cd = ds[["hour", "price", "distance"]].dropna().copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cd)
        km = KMeans(n_clusters=4, random_state=42)
        cd["cluster"] = km.fit_predict(X_scaled)

        cluster_labels = {
            0: "🌆 Evening Short Rides",
            1: "🌅 Early Morning Short",
            2: "💼 Afternoon Long Expensive",
            3: "🏙️ Morning Long Cheap"
        }
        cd["Cluster Type"] = cd["cluster"].map(cluster_labels)

        # Plot 1 — Hour vs Price
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        for idx, (label, group) in enumerate(cd.groupby("Cluster Type")):
            ax.scatter(group["hour"], group["price"],
                       label=label, alpha=0.4, s=10, color=colors[idx])
        ax.set_title("Ride Clusters — Hour vs Price", fontsize=14)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig)

        # Plot 2 — Distance vs Price
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for idx, (label, group) in enumerate(cd.groupby("Cluster Type")):
            ax2.scatter(group["distance"], group["price"],
                        label=label, alpha=0.4, s=10, color=colors[idx])
        ax2.set_title("Ride Clusters — Distance vs Price", fontsize=14)
        ax2.set_xlabel("Distance (miles)")
        ax2.set_ylabel("Price ($)")
        ax2.legend(loc="upper right", fontsize=8)
        st.pyplot(fig2)

        # Summary table
        st.markdown("#### Cluster Summary")
        summary = cd.groupby("Cluster Type")[["hour", "price", "distance"]].mean().round(2)
        st.dataframe(summary)
        st.info("💡 Each color = one ride pattern group. Orange cluster = highest value rides")
        st.markdown("---")

    # ── Step 10 — Price Predictor ──
    if st.session_state["run_10"]:
        st.markdown("**📌 Step 10 — 🎯 Price Predictor**")

        # Train model silently
        ds2 = ds.copy()
        ds2["expensive"] = (ds2["price"] > ds2["price"].mean()).astype(int)
        X = ds2[["hour", "distance", "surge_multiplier"]].dropna()
        y = ds2.loc[X.index, "expensive"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        st.markdown("#### Enter Ride Details")
        col1, col2 = st.columns(2)
        with col1:
            hour_input = st.slider("🕐 Hour of Day", 0, 23, 12, key="pred_hour")
            distance_input = st.number_input("📍 Distance (miles)", 0.5, 10.0, 2.0, key="pred_dist")
        with col2:
            surge_input = st.selectbox("⚡ Surge Multiplier", [1.0, 1.25, 1.5, 2.0], key="pred_surge")
            cab_input = st.selectbox("🚗 Cab Type", ["Uber", "Lyft"], key="pred_cab")

        if st.button("🔮 Predict Now"):
            prediction = model.predict([[hour_input, distance_input, surge_input]])[0]
            prob = model.predict_proba([[hour_input, distance_input, surge_input]])[0]

            if prediction == 1:
                st.error(f"💸 This ride is likely **EXPENSIVE** (above ${ds['price'].mean():.2f} avg)")
            else:
                st.success(f"✅ This ride is likely **CHEAP** (below ${ds['price'].mean():.2f} avg)")

            st.metric("Confidence", f"{max(prob) * 100:.1f}%")

            st.markdown("#### 📊 Your Input vs Dataset Average")
            factors = pd.DataFrame({
                "Factor": ["Hour", "Distance (miles)", "Surge Multiplier"],
                "Your Input": [hour_input, distance_input, surge_input],
                "Dataset Average": [
                    round(ds["hour"].mean(), 1),
                    round(ds["distance"].mean(), 2),
                    round(ds["surge_multiplier"].mean(), 2)
                ]
            })
            st.dataframe(factors)
        st.markdown("---")
