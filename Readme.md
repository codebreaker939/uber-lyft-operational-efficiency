🚕 Uber vs Lyft Operational Efficiency Analysis (Boston)
📌 Problem Statement

To compare the operational efficiency of Uber and Lyft in Boston using real ride data, focusing on cost comparison and demand clustering to identify pricing behavior and demand patterns across time and locations.

📂 Dataset

Source: Kaggle

Dataset: Uber and Lyft Dataset – Boston, MA

Description:
The dataset contains ride-level information such as price, distance, time, ride type, surge multiplier, and geographical coordinates for Uber and Lyft rides in Boston.

🛠️ Tech Stack Used

Programming Language: Python

Libraries:

Pandas, NumPy – Data manipulation

Matplotlib, Seaborn – Data visualization

Scikit-learn – Clustering and scaling

Environment: Jupyter Notebook

🔍 Project Approach
1️⃣ Data Preprocessing

Removed missing and inconsistent values

Filtered rides specific to Boston

Extracted time-based features such as hour of the day

Created derived metrics like price per mile for efficiency comparison

2️⃣ Cost Comparison Analysis

The following cost metrics were analyzed for Uber and Lyft:

Average ride price

Price per mile (cost efficiency)

Surge multiplier comparison

Price variation across different hours of the day

Ride-type-wise cost comparison

Identification of peak and off-peak pricing behavior

3️⃣ Demand Analysis

Analyzed ride demand distribution by hour

Identified peak demand hours

Studied popular pickup and drop-off locations

Compared demand patterns between Uber and Lyft

4️⃣ Demand Clustering

Applied K-Means clustering to identify demand patterns

Features used for clustering:

Hour of day

Ride price

Ride distance

Standardized data using StandardScaler

Visualized clusters to interpret high-demand and high-cost periods

📊 Key Findings

Uber generally shows lower price per mile, indicating better cost efficiency in several scenarios

Lyft exhibits higher surge sensitivity during peak hours

Peak demand occurs during typical commute hours

Clustering reveals distinct ride patterns such as:

High-demand, high-cost peak-hour rides

Low-demand, low-cost off-peak rides

✅ Conclusion

The analysis demonstrates that both Uber and Lyft have distinct operational strategies. While Uber appears more cost-efficient on average, Lyft’s pricing fluctuates more during peak demand periods. Demand clustering highlights critical operational stress periods that influence pricing and availability for both services.

🚀 Future Enhancements

Use advanced clustering techniques like DBSCAN

Incorporate weather and traffic data

Perform real-time demand prediction

Extend analysis to other cities

📁 Project Structure
Internal_Project_Business/
│
├── main.ipynb
├── data/
│   └── uber_lyft_boston.csv
├── results/
├── README.md
└── requirements.txt
📌 How to Run the Project

Clone or download the project

Install required dependencies

pip install -r requirements.txt

Open main.ipynb in Jupyter Notebook

Run cells sequentially