from sklearn.cluster import KMeans
from kneed import KneeLocator

# compute inertia for a range of k
ks = range(1, 11)
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

# detect elbow
knee = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
optimal_k = knee.knee
print("Elbow at k =", optimal_k)
