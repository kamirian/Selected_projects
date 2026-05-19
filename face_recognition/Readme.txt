Each classifier is implemented in its own dedicated Jupyter notebook, which can be executed from top to bottom. All notebooks include the necessary compute_pca and compute_mda functions, so they are fully self-contained.

A separate notebook is provided specifically for visualizing the PCA and MDA projections of the original dataset—this is intended for illustration purposes only and is not tied to any specific classifier.

For the AdaBoost and kernel SVM experiments, two versions are implemented:

One using gradient descent to solve the SVM dual problem.

Another using CVXOPT for quadratic programming.

While both approaches were tested, the results for AdaBoost with CVXOPT-based SVM were not included in the final report to keep it concise. Additionally, CVXOPT showed numerical instability when used with higher-degree polynomial kernels, which further motivated the choice of the gradient descent-based SVM as the primary implementation throughout the report.