from preprocess import load_nsl_kdd, split_xy, build_preprocessor
from imblearn.over_sampling import SMOTE

train, test = load_nsl_kdd()
print(train.shape, test.shape)
print(train["attack_category"].value_counts().to_dict())
print(test["attack_category"].value_counts().to_dict())

X_tr, y_tr = split_xy(train, "attack_category")
X_te, y_te = split_xy(test, "attack_category")
pre = build_preprocessor(X_tr)
A = pre.fit_transform(X_tr)       # fit on TRAIN only
B = pre.transform(X_te)           # test: transform only
print("features after encoding:", A.shape[1], A.shape, B.shape)
Xs, ys = SMOTE(random_state=42).fit_resample(A, y_tr)   # train only
print("after SMOTE:", ys.value_counts().to_dict())