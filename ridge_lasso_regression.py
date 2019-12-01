import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
le = preprocessing.LabelEncoder()

credit = pd.read_csv('Credit.csv')

le.fit(credit['Gender'])
gender_val = le.transform(credit['Gender'])

le.fit(credit['Student'])
stu_val = le.transform(credit['Student'])

le.fit(credit['Married'])
ma_val = le.transform(credit['Married'])

le.fit(credit['Ethnicity'])
eth_val = le.transform(credit['Ethnicity'])

train_data = np.hstack((credit.values[:, 1:4], stu_val.reshape((-1, 1)),
            credit.values[:, 4:7], gender_val.reshape((-1, 1)),
            ma_val.reshape((-1, 1)), eth_val.reshape((-1, 1))))

balance = credit.values[:, -1].reshape((-1, 1))
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression

alpha_seq = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.5, 2, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 2000, 5000, 10000]
coe_val = np.zeros((len(alpha_seq), train_data.shape[1]))

train_data = train_data/np.std(train_data.astype(float), axis=0).reshape((1, -1))
legend_label_seq = ['Income', 'Limit', 'Rating', 'Student']
for alpha_index, alpha_val in enumerate(alpha_seq):
    coe_val[alpha_index] = ridge_regression(train_data, balance, alpha = alpha_val)


fontsizess = 15
for alpha_index in range(train_data.shape[1]):
    if alpha_index<=3:
        plt.plot(np.log10(alpha_seq), coe_val[:, alpha_index], label = legend_label_seq[alpha_index])
    else:
        plt.plot(np.log10(alpha_seq), coe_val[:, alpha_index], c='grey')
plt.plot([2, 2], [-200, 600], linestyle = 'dotted')
plt.legend(frameon=False, fontsize = fontsizess)
plt.ylabel('Coefficient values', fontsize = fontsizess)
plt.xlabel(r'$\lambda$', fontsize = fontsizess)
plt.xticks([-2, -1, 0, 1, 2, 3, 4], [r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'], fontsize = fontsizess)
plt.tight_layout()
plt.savefig('ridge_coeff_1.pdf', bbox_inches='tight')
plt.show()


coe_val = np.zeros((len(alpha_seq), train_data.shape[1]))

for alpha_index, alpha_val in enumerate(alpha_seq):
    lasso = Lasso(alpha=alpha_val)
    lasso.fit(train_data.astype(float), balance)
    coe_val[alpha_index] = lasso.coef_

for alpha_index in range(train_data.shape[1]):
    if alpha_index<=3:
        plt.plot(np.log10(alpha_seq), coe_val[:, alpha_index], label = legend_label_seq[alpha_index])
    else:
        plt.plot(np.log10(alpha_seq), coe_val[:, alpha_index], c='grey')
plt.plot([2, 2], [-200, 400], linestyle = 'dotted')
plt.legend()
plt.legend(frameon=False, fontsize = fontsizess)
plt.xlabel(r'$\lambda$', fontsize = fontsizess)
plt.ylabel('Coefficient values', fontsize = fontsizess)
plt.xticks([-2, -1, 0, 1, 2, 3, 4], [r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'], fontsize = fontsizess)
plt.tight_layout()
plt.savefig('lasso_coeff_2.pdf', bbox_inches='tight')
plt.show()

