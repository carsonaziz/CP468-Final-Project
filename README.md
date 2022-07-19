# Cancer Type Classification
This project took the Gene Expression Dataset (Golub et al.) and ran machine learning algorithms to attempt to identify patterns in the dataset and classify cancer types. The objective is to identify a patients cancer type from data based on gene expression. The results we achieved were very close to the actual patient cancer types with Cluster 1 attracting 48 patients, and Cluster 2 attracting 24 patients, while the actual patient cancer distribution had 47 ALL cancer types and 25 AML cancer types.

## Dataset
The datasets used in this project were provided alongside the Golub et al. 1999 paper. Which consisted of three seperate datasets, a dataset that mapped the patient to their actual cancer type (actual.csv), a dataset for training which contained gene expression data (data_set_ALL_AML_train.csv), and a dataset for testing which contained gene expression data as well (data_set_ALL_AML_independant.csv)

## Installation & Execution
Follow the steps below to run the program:
1. Navigate to your desired directory and clone the repository using `git clone https://github.com/carsonaziz/CP468-Final-Project.git`
2. To install the necessary libraries, run `!pip install pandas` and `!pip install matplotlib` from the repository directory
3. To run the program use `python3 main.py`

## Performance Parameters
To measure performace, we compared the actual cluster sizes provided in the actual.csv dataset to the cluster sizes that were observed after running the machine learning algorithms on the training and testing datasets.

## About Developers
Carson Aziz - Wilfrid Laurier computer science student. Age 21.
*Insert info

## License
No license associated with this repository