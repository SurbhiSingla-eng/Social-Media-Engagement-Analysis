import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. DATA ACQUISITION & FEATURE ENGINEERING (Unit I)
# ---------------------------------------------------------
print("=== 1. Preprocessing & Feature Engineering ===")

# Create dummy data matching your Google Sheet's proportions
np.random.seed(42)
n_rows = 5000
platforms = ['TikTok', 'Instagram', 'Twitter', 'YouTube']

data = {
    'Post_ID': [f'Post_{i+1}' for i in range(n_rows)],
    'Platform': np.random.choice(platforms, size=n_rows),
    'Views': np.random.randint(1000, 5000000, size=n_rows),
    'Likes': np.random.randint(500, 500000, size=n_rows),
    'Shares': np.random.randint(50, 100000, size=n_rows),
    'Comments': np.random.randint(10, 50000, size=n_rows),
    'Date': pd.date_range(start='2026-01-01', periods=n_rows, freq='h')
}

df = pd.DataFrame(data)

# Feature Engineering
df['Total_Interactions'] = df['Likes'] + df['Shares'] + df['Comments']
df['Engagement_Rate'] = df['Total_Interactions'] / df['Views']

# Feature Scaling
scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()

df['Views_Normalized'] = scaler_minmax.fit_transform(df[['Views']])
df['Views_Standardized'] = scaler_std.fit_transform(df[['Views']])

print(df[['Post_ID', 'Platform', 'Total_Interactions', 'Engagement_Rate']].head())
print("\nNormalization & Standardization successfully applied.")


# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA) (Unit II)
# ---------------------------------------------------------
print("\n=== 2. Descriptive Statistics & EDA ===")
desc_stats = df[['Views', 'Likes', 'Shares', 'Comments']].agg(['mean', 'median', 'std', 'skew', 'kurtosis']).T
desc_stats.columns = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
print(desc_stats)

print("\n--- Correlation Matrix ---")
corr_matrix = df[['Views', 'Likes', 'Shares', 'Comments']].corr()
print(corr_matrix)


# ---------------------------------------------------------
# 3. INFERENTIAL STATISTICS & SAMPLING (Unit III)
# ---------------------------------------------------------
print("\n=== 3. Inferential Statistics ===")

# Stratified Sampling
stratified_sample = df.groupby('Platform', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
print(f"Stratified sample size: {len(stratified_sample)} rows")

# T-Test (Comparing Likes of TikTok vs Instagram)
tiktok_likes = df[df['Platform'] == 'TikTok']['Likes']
insta_likes = df[df['Platform'] == 'Instagram']['Likes']
t_stat, t_pval = stats.ttest_ind(tiktok_likes, insta_likes)
print(f"T-Test p-value: {t_pval:.4f}")

# ANOVA (Testing if views differ across platforms)
model_anova = ols('Views ~ C(Platform)', data=df).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print("\n--- ANOVA Results ---")
print(anova_table)

# Chi-Square Test (Platform vs high engagement)
df['High_Engagement'] = df['Engagement_Rate'] > df['Engagement_Rate'].median()
contingency_table = pd.crosstab(df['Platform'], df['High_Engagement'])
chi2_stat, chi2_pval, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test p-value: {chi2_pval:.4f}")


# ---------------------------------------------------------
# 4. REGRESSION & PREDICTIVE MODELING (Unit IV)
# ---------------------------------------------------------
print("\n=== 4. Regression & Predictive Modeling ===")

# Multiple Linear Regression (Predicting Views using Likes, Shares, Comments)
X_reg = df[['Likes', 'Shares', 'Comments']]
y_reg = df['Views']
reg_model = LinearRegression().fit(X_reg, y_reg)
print(f"Linear Regression R² Score: {reg_model.score(X_reg, y_reg):.4f}")

# Logistic Regression (Predicting High Engagement)
X_log = df[['Views', 'Likes', 'Shares']]
y_log = df['High_Engagement'].astype(int)
log_model = LogisticRegression().fit(X_log, y_log)
print(f"Logistic Regression Accuracy: {log_model.score(X_log, y_log):.4f}")


# ---------------------------------------------------------
# 5. MULTIVARIATE ANALYSIS & CLUSTERING (Unit V)
# ---------------------------------------------------------
print("\n=== 5. Multivariate Analysis ===")

# Principal Component Analysis (PCA)
features = df[['Views', 'Likes', 'Shares', 'Comments']]
scaled_features = scaler_std.fit_transform(features)
pca = PCA(n_components=2)
pca_results = pca.fit_transform(scaled_features)
print(f"Explained Variance Ratio by top 2 PCA components: {pca.explained_variance_ratio_}")

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(scaled_features)
print(f"Clustering completed. Assigned clusters for first 5 rows:\n{df['Cluster'].head()}")


# ---------------------------------------------------------
# 6. TIME-SERIES ANALYSIS (Unit VI)
# ---------------------------------------------------------
print("\n=== 6. Time-Series Analysis ===")

# Setup time-series data
ts_data = df.set_index('Date')['Views'].resample('D').mean().fillna(method='ffill')

# Augmented Dickey-Fuller Test (Stationarity Check)
adf_result = adfuller(ts_data)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f} (Stationary if p < 0.05)")

# Fit a simple ARIMA model
arima_model = ARIMA(ts_data, order=(1, 1, 1))
fitted_arima = arima_model.fit()
print("\n--- ARIMA Model Summary ---")
print(fitted_arima.summary().tables[1])

print("\n=== Pipeline Execution Completed Successfully! ===")
