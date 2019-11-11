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

# Helper function to augment table data
def graph_of_thrones(G1, name):
    if G1.nodes[name]:
        return [G1.nodes[name]['betweenness'], G1.nodes[name]['community'], G1.nodes[name]['rank'], G1.nodes[name]['degree']]
    return [None, None, None, None]

def pipeline(df, names, classifiers, weight=None):
    book1_df = pd.read_csv('data/asoiaf-book1-edges.csv')
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
        sorted(relationships, key=(lambda x: x[0]), reverse=False)[:10:2]
        # Plot distribution of weights
        plt.hist(wts);
        plt.show()

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
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        print(f'Base F1 Score for {name}: {f1_score(y_test, clf.predict(X_test))}')
        clf.fit(GX_train, Gy_train)
        print(f'Graph Augmented F1 Score for {name}: {f1_score(Gy_test, clf.predict(GX_test))} \n')

