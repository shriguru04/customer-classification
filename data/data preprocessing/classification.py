
python3 << 'EOF'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                              precision_score, recall_score, silhouette_score)
import warnings, json
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/tanma_tkiisqy/Desktop/chakre sir mini project/data/beauty customers.csv")

le_gender   = LabelEncoder()
le_category = LabelEncoder()
df['Gender_enc']   = le_gender.fit_transform(df['Gender'])
df['Category_enc'] = le_category.fit_transform(df['Product_Category'])

feature_cols = ['Age','Income','Spending_Score','Purchase_Frequency',
                'Avg_Order_Value','Service_Usage','Last_Purchase_Days',
                'Gender_enc','Category_enc']

X_raw = df[feature_cols].values

# Impute BEFORE scaling
imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(X_raw)
print("NaN after impute:", np.isnan(X_imp).sum())

y = df['True_Segment'].values
le_seg = LabelEncoder()
y_enc  = le_seg.fit_transform(y)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# ANOMALY DETECTION
iso = IsolationForest(contamination=0.05, random_state=42)
anom_preds = iso.fit_predict(X_scaled)
df['Anomaly_Flag'] = anom_preds
anomalies = df[df['Anomaly_Flag'] == -1]
print(f"Anomalies: {len(anomalies)}")

# PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} components")
print("Variance explained:", np.round(pca.explained_variance_ratio_,3))

# BEST K
sil_scores = {}
for k in range(2, 8):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_pca)
    sil_scores[k] = round(silhouette_score(X_pca, lbl), 4)
best_k = max(sil_scores, key=sil_scores.get)
print(f"Silhouette scores: {sil_scores}  |  Best K={best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)
final_sil = silhouette_score(X_pca, cluster_labels)

# SPLIT
X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y_enc, test_size=0.2,
                                            random_state=42, stratify=y_enc)
print(f"Train={len(X_tr)} Test={len(X_te)}")

# RF
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)

acc  = accuracy_score(y_te, y_pred)
prec = precision_score(y_te, y_pred, average='weighted')
rec  = recall_score(y_te, y_pred, average='weighted')
f1   = f1_score(y_te, y_pred, average='weighted')
print(f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

# K-FOLD
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv  = cross_val_score(rf, X_pca, y_enc, cv=skf, scoring='accuracy')
print(f"CV: {np.round(cv,4)} | mean={cv.mean():.4f}")

cr = classification_report(y_te, y_pred, target_names=le_seg.classes_, output_dict=True)

# UNKNOWN PREDICTION
unknown_raw = np.array([[28, 45000, 75, 10, 2200, 4, 7, 1, 2]], dtype=float)
u_imp    = imputer.transform(unknown_raw)
u_scaled = scaler.transform(u_imp)
u_pca    = pca.transform(u_scaled)
pred_seg = le_seg.inverse_transform(rf.predict(u_pca))[0]
pred_proba = rf.predict_proba(u_pca)[0]
anom_flag  = iso.predict(u_scaled)[0]

results = {
    'best_k': int(best_k),
    'silhouette': round(final_sil, 4),
    'sil_scores': sil_scores,
    'accuracy': round(acc, 4),
    'precision': round(prec, 4),
    'recall': round(rec, 4),
    'f1': round(f1, 4),
    'cv_scores': [round(s,4) for s in cv.tolist()],
    'cv_mean': round(cv.mean(), 4),
    'cv_std': round(cv.std(), 4),
    'n_anomalies': int(len(anomalies)),
    'pca_components': int(X_pca.shape[1]),
    'pca_variance': [round(v,4) for v in pca.explained_variance_ratio_.tolist()],
    'segments': le_seg.classes_.tolist(),

> Shriguru:
'anomaly_ids': anomalies['Customer_ID'].tolist()[:20],
    'class_report': cr,
    'unknown_pred': pred_seg,
    'unknown_anomaly': int(anom_flag),
    'unknown_proba': {cls: round(float(p),4) for cls, p in zip(le_seg.classes_, pred_proba)},
    'feature_importance': {feature_cols[i]: round(float(rf.feature_importances_[i]),4) 
                           for i in range(len(feature_cols))}
}
with open('/home/claude/results.json','w') as f:
    json.dump(results, f)
print("✅ ALL DONE")
EOF
