# KNN on Gene Expression Data

A small classification project that uses K Nearest Neighbors to predict whether cancer is present in a sample based on two gene expression measurements. The work lives in `KNN_Gene_Expression.ipynb` and walks through exploring the data, scaling the features, fitting a KNN model, choosing a good value of K, and then automating that choice with a cross validated grid search.

## The dataset

The file `gene_expression.csv` holds 3000 samples with three columns:

| Column | Meaning |
| --- | --- |
| Gene One | expression level of the first gene |
| Gene Two | expression level of the second gene |
| Cancer Present | target label, 1 if cancer is present and 0 if not |

The classes are perfectly balanced with 1500 positive and 1500 negative samples. Because there are only two features the data is easy to plot, and a scatter plot of Gene One against Gene Two shows two groups that are mostly separable with some overlap near the boundary. That overlap is exactly the situation where a distance based classifier like KNN tends to do well.

## The approach

KNN classifies a new point by looking at the labels of its K closest neighbors in the training set and taking a majority vote. The steps in the notebook follow that idea end to end:

1. Load the CSV and run quick exploratory plots (a scatter plot and a seaborn pairplot) to confirm the two classes are distinguishable.
2. Split the features `X` (Gene One, Gene Two) and the label `y` (Cancer Present), then split into train and test sets with a 33 percent test size and a fixed random seed.
3. Standardize the features with `StandardScaler`. Scaling matters for KNN because the algorithm depends on distances, so features on different ranges would otherwise dominate the vote.
4. Fit a `KNeighborsClassifier` and evaluate it on the held out test set.
5. Use the elbow method, sweeping K from 1 to 29 and plotting the test error rate, to see how the choice of K affects performance.
6. Replace the manual sweep with a `GridSearchCV` over a `Pipeline` that chains the scaler and the classifier, so the model selects K through five fold cross validation instead of by eye.

## Results

The test set contains 990 samples. A first model with K equal to 9 reached an accuracy of about 0.93. Its confusion matrix was:

```
[[432  30]
 [ 38 490]]
```

and the classification report showed precision and recall around 0.93 to 0.94 for both classes.

The elbow sweep confirmed that error stays low across a wide range of K, bottoming out near K equal to 9. When the grid search took over the selection through cross validation, it settled on K equal to 16 and produced a slightly stronger result, with accuracy of about 0.94 and precision, recall, and f1 all near 0.94 for both classes. The lesson the notebook draws out is to let the model pick K through cross validation rather than reading it off a plot, since the cross validated choice generalizes better.

## How to run it

Install the dependencies and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook KNN_Gene_Expression.ipynb
```

The notebook reads the CSV with an absolute path from the original machine. If you run it elsewhere, change the `pd.read_csv(...)` call near the top to point at the `gene_expression.csv` file in this repository:

```python
df = pd.read_csv('gene_expression.csv')
```

A standalone script version of the same workflow is included at `src/knn_gene_expression.py` for anyone who prefers to run it without Jupyter:

```bash
python src/knn_gene_expression.py
```

## Files

- `KNN_Gene_Expression.ipynb` the original notebook with the full analysis and plots
- `gene_expression.csv` the dataset
- `src/knn_gene_expression.py` a script that reproduces the modeling steps
- `requirements.txt` the Python dependencies
