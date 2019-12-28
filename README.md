# Genetic Algorithm on K-Means Clustering of PHD Programs with BM-25 Score

This is a fork from [Genetic-Algorithm-on-K-Means-Clustering](https://github.com/amirdeljouyi/Genetic-Algorithm-on-K-Means-Clustering) and was developed as a task for the Evolutive Systems class of 2019 for the Computer Science Masters Course at UFPel.

## Approach
PHD program clustering using Genetic K-Means and BM25 scores 
- Corpus of each PHD program made of the title and resume of all their indexed production at the official brazilian plataform [Sucupira](sucupira.capes.gov.br/) -- articles, books, presentations, thesis, and etc, with resume being available mostly for dissertation and thesis only
- Using ElasticSearch and the generated dataset, extracted the top 50 scoring keywords for each program with Okapi BM-25 
- For each unique term (around 11k), gets the score of that term for each program, generating a matrix
- Minmax normalization for standardization
- Davies–Bouldin index for evaluation of each cluster
- In Genetic
  * Rank based selection
  * One point crossover

## Requirements
- panda
- numpy

## Getting Started
```
python __main__.py
```

## Input
- ```config.txt``` contain control parameters
  * kmax : maximum number of clusters
  * budget : budget of how many times run GA
  * numOInd : number of Individual
  * Ps : probability of ranking Selection
  * Pc : probability of crossover
  * Pm : probability of mutation

## Output
- ```norm_data.csv``` is normalization data
- ```cluster_json``` is centroid of each cluster
- ```result.csv``` is data with labeled to each cluster
