# Re-produced co-SVD

## Dataset
The dataset used in this work is from the [MovieLens dataset](https://grouplens.org/datasets/movielens/). There are three sets of datasets obtained from [GroupLens](https://grouplens.org).


`ml-latest-small` MovieLens 100K dataset (Year 2016) (Uploaded to this repository)

`mlsmall`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
MovieLens 100K latest dataset (Year 2018) (Uploaded to this repository)

`ml10M100K` MovieLens 10M Dataset (Link: https://grouplens.org/datasets/movielens/10m/)

## Installation

The Python version of this work is 3.7

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [Surprise package](http://surpriselib.com/).

```bash
pip install surprise
```

This work is also required [cython](https://cython.org/)

After the package installed, the folder of the (surprise) package need to be replaced with the surprise folder provided in this repository.

## Usage

To initial the CoSVD
```python
tags = pd.read_csv('path\to\tags.csv')
algo = CoSVD(verbose=False, n_epochs=65, lr_all=0.0028, n_factors=40, tags=tags, random_state=123)
```

Model Training & Evaluation
```python
## Model Training
algo.fit(trainset)

## Model Testing
predictions = algo.test(testset)

## Model Evaluation
mae = accuracy.mae(predictions, verbose=False)
rmse = accuracy.rmse(predictions, verbose=False)
```
This work utilized Surprise package to build the own prediction algorithm (co-SVD). So, this work supports most of the features provided by Surprise package. For more information, you may check the Surprise package [documentation](https://surprise.readthedocs.io/en/stable/).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
