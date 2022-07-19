from copy import deepcopy
import math
import pandas as pd
import matplotlib.pyplot as plt


# calculate euclidean distance between point_a and point_b
def euclid_distance(point_a, point_b) -> float:
    sum = 0
    for i in range(len(point_a)):
        sum += (point_a[i] - point_b[i]) * (point_a[i] - point_b[i])
    
    return float(math.sqrt(sum))


# k-means algorithm for n lnegth data
def kmeans(k: int, data) -> tuple:
    clusters = []
    centroids = []

    # Assign initial centroids as first k patients
    for i in range(k): centroids.append(deepcopy(data[i]))

    # clusters[i] value represents the cluster i belongs to, initalize all to 0
    for _ in range(len(data)): clusters.append(0)

    # assigns data points to appropriate centroid
    for j in range(len(data)):
        # sets initial centroid to centroid at 0 index of centroids
        closest_cent_dist = euclid_distance(data[j], centroids[0])

        for i in range(len(centroids)):
            # calulates euclidean distance from data point to each centroid and sets centroid for that data point to closest centroid
            new_distance = euclid_distance(data[j], centroids[i])   

            if new_distance <= closest_cent_dist:
                closest_cent_dist = new_distance
                clusters[j] = i

    # Calculates new positions of centroids by centering them within their cluster and updates the centroid each data point is assigned to
    # Stops when each centroid converges to a single position
    centroids_changed = True
    while centroids_changed:
        centroids_changed = False
        # calculate new centroids
        for j in range(len(centroids)):
            sums = [0] * len(data[0])
            sum_x = 0
            sum_y = 0
            count = 0
            for i in range(len(clusters)):
                if clusters[i] == j:        # clusters[i] is the cluster i belongs to, j represents one of the clusters
                    for dim_i in range(len(data[0])):
                        sums[dim_i] += deepcopy(data[i][dim_i])
                        count += 1
            
            # if no data points are closest to centroid skip to next centroid
            if count == 0: continue

            # calculate new centroid position
            new_cent_coord = [0] * len(data[0])
            for dim_i in range(len(data[0])):
                new_cent_coord[dim_i] = sums[dim_i] / count


            # if any dimension of new centroid position is different from previous position, set centroids_changed to True (to keep the loop going)
            # and update centroid position
            for dim_i in range(len(data[0])):
                if new_cent_coord[dim_i] != centroids[j][dim_i]:
                    centroids[j][dim_i] = new_cent_coord[dim_i]
                    centroids_changed = True

        # reassign data points to closest centroid
        for j in range(len(data)):
            closest_cent_dist = euclid_distance(data[j], centroids[0])
            for i in range(len(centroids)):
                new_distance = euclid_distance(data[j], centroids[i])
                if new_distance <= closest_cent_dist:
                    closest_cent_dist = new_distance
                    clusters[j] = i

    return clusters, pd.array(centroids, dtype=int)


if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    result_data = pd.read_csv("dataset/actual.csv")
    train_data = pd.read_csv("dataset/data_set_ALL_AML_train.csv")
    test_data = pd.read_csv("dataset/data_set_ALL_AML_independent.csv")

    #################################################################################
    # Process Data ##################################################################
    #################################################################################

    # Result Data ################################################
    result_data["patient"] = result_data["patient"].astype("int")



    # Training Data ##############################################
    # rename call columns
    for col in train_data.columns:
        if "call" in col:
            index = train_data.columns.get_loc(col)
            patient = train_data.columns[index-1]
            train_data.rename(columns={col: f'call{patient}'}, inplace=True)

    
    # Merge "Gene Description" and "Gene Accession Number" to make them one piece of data to represent the gene
    train_data["Gene"] = train_data["Gene Description"] + '_' + train_data["Gene Accession Number"]

    # Transpose dataframe
    train_data = train_data.T
    train_data.columns = train_data.iloc[-1]
    train_data = train_data[2:-1]



    # Test Data ##################################################
    # rename call columns
    for col in test_data.columns:
        if "call" in col:
            index = test_data.columns.get_loc(col)
            patient = test_data.columns[index-1]
            test_data.rename(columns={col: f'call{patient}'}, inplace=True)

    # Merge "Gene Description" and "Gene Accession Number" to make them one piece of data to represent the gene
    test_data["Gene"] = test_data["Gene Description"] + '_' + test_data["Gene Accession Number"]

    # Transpose dataframe
    test_data = test_data.T
    test_data.columns = test_data.iloc[-1]
    test_data = test_data[2:-1]



    # Merged Data ################################################
    # Merge test and train datasets
    data = pd.concat([train_data, test_data], axis=0, join="inner", sort=False)

    # Remove genes with absent(A) detection call in every patient
    call_rows = [row for row in data.index if "call" in row]                      # rows to check for value "A"
    conditional = data.filter(call_rows, axis=0).apply(lambda x: x == 'A', axis=1).all()
    data = data.loc[:, ~conditional]

    # Remove call rows
    data = data.drop(call_rows, axis=0)

    # Add patient label
    data["patient"] = data.index
    data["patient"] = data["patient"].astype("int")
    
    # Attach cancer classification to patients
    data = pd.merge(left=data, right=result_data, left_on='patient', right_on="patient")

    # Extract patient data (only gene data)
    patient_data = data.iloc[:, 0:-2]
    X = patient_data.values




    #################################################################################
    # Analyze Data ##################################################################
    #################################################################################
    # Apply k-means algorithm to data
    clusters, centroids = kmeans(2, X)
    print("Cluster 1: " + str(clusters.count(1)) + "\n" + "Cluster 2: " + str(clusters.count(0)))
    print(result_data["cancer"].value_counts())




    #################################################################################
    # Plot Data #####################################################################
    #################################################################################

    plt.rcParams["figure.figsize"] = (10, 8)

    # Plot actual cancer classifications
    plt.subplot(1, 2, 1)
    plt.bar(result_data["cancer"].value_counts().index, result_data["cancer"].value_counts())
    plt.title("Cancer Classification Distribution (ALL | AML)")

    # Plot results
    plt.subplot(1, 2, 2)
    x_labels = ["Cluster 1", "Cluster 2"]
    plt.bar(x_labels, [clusters.count(1), clusters.count(0)])
    plt.title("Learned Cancer Classification | K-Means\n (ALL | AML)")
    plt.show()