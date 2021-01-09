# Re-produced co-SVD

## Dataset
The dataset used in this work is from the [MovieLens dataset](https://grouplens.org/datasets/movielens/). There are three sets of datasets obtained from [GroupLens](https://grouplens.org).


`ml-latest-small` MovieLens 100K dataset (Year 2016) (Uploaded to this repository)

`mlsmall`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
MovieLens 100K latest dataset (Year 2018) (Uploaded to this repository)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [Surprise package](http://surpriselib.com/).

```bash
pip install surprise
```

This work is also required [cython](https://cython.org/)

## Usage
To generate the User-Tag Matrix & Item-Tag Matrix as the work proposed by [Luo et al.](https://www.sciencedirect.com/science/article/abs/pii/S0957417418307231)
```python
# 'p_ut' is User-Tag Matrix
# 'f_it' is Item-Tag Matrix
# 'tags' is the tags data with tag internal id
# 'ratings' is a joined table of rating data and tag data

import matrices_generation as mg
p_ut, f_it, tags, ratings = mg.generateTagsOrigin(rate, raw_tags)
```

To initial the co-SVD
```python
algo = co_SVD(n_epochs=40, lr_all=0.006, n_factors=40
              , p_ut=p_ut, f_it=f_it, tags=tags, ratings=ratings)
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
