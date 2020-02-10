# Base imports
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# Classifier imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def graph_of_thrones(G1, name):
    """ Helper function to augment tabular data.
    Parameters:
            G1 (Graph Network): Collection of nodes.
            name (String): Character's name to find in the network.
    Returns:
        [features]: List of to be appended to DataFrame, or if no match then a List of None values.
    """
    if G1.nodes[name]:
        return [G1.nodes[name]['betweenness'], G1.nodes[name]['community'], G1.nodes[name]['rank'], G1.nodes[name]['degree']]
    return [None, None, None, None]

def pipeline(df, names, classifiers, weight=None):
    """ Distill visualization of weights and creation of models into a single pipeline.
    Parameters:
            df (Pandas DataFrame): Initial tabular data to be augmented.
            names ([Strings]): List of model names to match with below classifiers.
            classifiers([Sklearn Models]): List of SK Learn models to train.
            weight (String, Default: None): Which edge weight to apply to the graph network.
    Returns:
        None: Function prints out the F1 score for each model both with and without the network augmented data.
    """
    # Read in tabular data
    book1_df = pd.read_csv('data/asoiaf-book1-edges.csv')

    # Create and populate graph network
    G1 = nx.Graph()
    for row in book1_df.iterrows():
        if weight == 'inverse':
            G1.add_edge(row[1]['Source'].replace('-', ' '), row[1]['Target'].replace('-', ' '),
                        weight=1 / row[1]['weight'])
        else:
            G1.add_edge(row[1]['Source'].replace('-', ' '), row[1]['Target'].replace('-', ' '), weight=row[1]['weight'])
    if weight:
        weight = 'weight'
    for name, degree in nx.degree_centrality(G1).items():
        G1.nodes[name]['degree'] = degree
    for label, community in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(G1)):
        for name in community:
            G1.nodes[name]['community'] = label
    for name, betweenness in nx.betweenness_centrality(G1, weight=weight, seed=42).items():
        G1.nodes[name]['betweenness'] = betweenness
    for name, rank in nx.pagerank(G1, weight=weight).items():
        G1.nodes[name]['rank'] = rank
    if weight:
        wts = []
        relationships = []
        # Loop through each node and its adjacent nodes and capture its weight
        for n, nbrs in G1.adj.items():
            for nbr, eattr in nbrs.items():
                wt = eattr['weight']
                wts.append(wt)
                relationships.append((wt, n, nbr))
        # Examine weighted relationship, skip every other one (undirected relationship)
        print(sorted(relationships, key=(lambda x: x[0]), reverse=True)[:10:2])
        # Plot distribution of weights
        plt.hist(wts);
        plt.show()

    # Create augmented DataFrame
    graph_df = pd.concat([df, pd.DataFrame(df.apply(lambda x: graph_of_thrones(G1, x.Name), axis=1).values.tolist(),
                                           columns=['Betweenness', 'Community', 'Rank', 'Degree'])], axis=1)
    augmented_df = pd.concat([graph_df.drop('Community', axis=1),
                              pd.get_dummies(graph_df['Community'], prefix='community', drop_first=True)], axis=1)
    # Tabular Data
    X, y = df.drop(['Name', 'Dead'], axis=1), df['Dead']
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Augmented Data
    X, y = augmented_df.drop(['Name', 'Dead'], axis=1), augmented_df['Dead']
    GX_train, GX_test, Gy_train, Gy_test = \
        train_test_split(X, y, test_size=.2, random_state=42)
    g_scaler = StandardScaler()
    GX_train = g_scaler.fit_transform(GX_train)
    GX_test = g_scaler.transform(GX_test)

    # Print out results for each model and DataFrame
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        print(f'Base F1 Score for {name}: {f1_score(y_test, clf.predict(X_test))}')
        clf.fit(GX_train, Gy_train)
        print(f'Graph Augmented F1 Score for {name}: {f1_score(Gy_test, clf.predict(GX_test))} \n')

