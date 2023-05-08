# Collaborative Filtering Recommender System Algorithms

## Description
This project focuses on the development and evaluation of recommender systems for collaborative filtering. The project is divided into four assignments, each implementing different algorithms and techniques to improve recommendation accuracy. The algorithms employed are as follows:

## Assignment 1: User and Item-Based Recommender with KNN
In this phase, a user and item-based recommender system was implemented using the K-Nearest Neighbors (KNN) algorithm. The objective was to provide personalized recommendations by finding similar users or items based on their ratings. Additionally, variance weighting and significance weighting techniques were incorporated to enhance the recommendation quality. The NMAE (Normalized Mean Absolute Error) achieved ranged from 0.56 to 0.7, indicating moderate accuracy.

## Assignment 2: Alternating Least Squares (ALS) Algorithm
The second assignment involved implementing the Alternating Least Squares (ALS) algorithm, a more complex and advanced technique. ALS is a matrix factorization method commonly used in collaborative filtering. It alternates between updating user and item latent factors to minimize the reconstruction error. The ALS algorithm demonstrated improved accuracy, with NMAE values ranging from 0.17 to 0.19, surpassing the performance of the KNN-based recommender.

## Assignment 3: Nuclear Norm Minimization (NNM)
For the third assignment, the focus shifted to Nuclear Norm Minimization (NNM) for matrix completion. NNM aims to estimate missing values in a sparse matrix by finding a low-rank matrix that fits the observed entries. This technique was applied to a more sparse matrix, resulting in an NMAE of approximately 0.2. The successful performance indicates the effectiveness of NNM, particularly for extremely sparse matrices.

## Assignment 4: Schatten-p Norm Algorithm
The final assignment involved implementing a research paper that proposed the use of the Schatten-p norm algorithm for collaborative filtering. This algorithm specifically addresses the challenges posed by extremely sparse matrices. By leveraging the Schatten-p norm, which promotes low-rank solutions, the algorithm provided accurate recommendations for the sparse matrix. The NMAE results demonstrated its effectiveness in handling such scenarios.

## Conclusion
This project showcases the development and evaluation of various recommender systems for collaborative filtering. Each assignment represents a different algorithmic approach, offering insights into the strengths and limitations of each technique. The iterative progression from a basic KNN-based approach to more advanced algorithms like ALS, NNM, and Schatten-p norm demonstrates the project's exploration of increasingly sophisticated techniques to improve recommendation accuracy.

## Dataset Description
The dataset used for this project is the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1682 movies. The ratings are on a scale of 1 to 5, with 1 being the lowest and 5 being the highest. The dataset is split into training and test sets, with 80% of the ratings in the training set and 20% in the test set. The training set is used to train the recommender system, while the test set is used to evaluate the system's performance.
[Dataset Link](https://grouplens.org/datasets/movielens/100k/)

## License
[MIT License](https://opensource.org/licenses/MIT)
