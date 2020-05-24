"""
===========================================================
Project: PHOTON Graph
===========================================================
Description
-----------
A wrapper containing functions for extracting graph measures that can then be
used for further machine learning analyses

Version
-------
Created:        09-07-2019
Last updated:   05-08-2019


Author
------
Vincent Holstein
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""

#TODO: create a few more local graph transformation functions
#TODO: make error messages for possible errors
#TODO: make documentation for every single method

import networkx
from networkx.algorithms import approximation
from networkx.algorithms.assortativity import average_neighbor_degree, average_degree_connectivity
from networkx.algorithms.centrality import degree_centrality, eigenvector_centrality, katz_centrality, closeness_centrality, current_flow_closeness_centrality, information_centrality
from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality, communicability_betweenness_centrality, load_centrality
from networkx.algorithms.centrality import edge_load_centrality
from networkx.algorithms.efficiency import global_efficiency
from networkx.algorithms.centrality.reaching import global_reaching_centrality
from networkx.algorithms.smallworld import sigma, omega
from networkx.algorithms.wiener import wiener_index
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import os

class GraphMeasureTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"
    measures_dict = {"global_efficiency": global_efficiency, 'sigma': sigma}

    def __init__(self, global_efficiency = 0, global_reaching_centrality = 0,
                 sigma = 0, omega = 0,
                 estrada_index = 0, s_metric = 0, triadic_census = 0,
                 wiener_index = 0, node_connectivity = 0,
                 max_clique = 0, average_clustering = 0,
                 metric_closure = 0, ramsey_number = 0,
                 min_egde_dominating_set = 0, degree_assortativity_coefficient = 0,
                 degree_pearson_correlation_coefficient = 0,
                 average_degree_connectivity = 0, k_nearest_neighbors = 0,
                 graph_clique_number = 0, transivity = 0,
                 communicability = 0, number_connected_components = 0,
                 number_strongly_connected_components = 0, number_weakly_connected_components = 0,
                 number_attracting_components = 0, overall_reciprocity = 0,
                 rich_club_coefficient = 0, average_shortest_path_length = 0,
                 current_flow_closeness_centrality = 0, information_centrality = 0,
                 betweeness_centrality = 0, k_nodes_betweenness_centrality = None,
                 edge_betweenness_centrality = 0, k_nodes_edge_betweenness_centrality = None,
                 load_centrality = 0, edge_load_centrality = 0,
                 communicability_betweenness_centrality = 0,
                 all_pairs_node_connectivity=0, all_pairs_node_connectivity_nbunch=None,
                 all_pairs_node_connectivity_cutoff=None,
                 average_neighbor_degree=0, degree_centrality=0,
                 eigenvector_centrality=0, katz_centrality=0,
                 closeness_centrality=0,
                 logs=''):
        self.global_efficiency = global_efficiency
        self.global_reaching_centrality = global_reaching_centrality
        self.sigma = sigma
        self.omega = omega
        self.estrada_index = estrada_index
        self.s_metric = s_metric
        self.triadic_census = triadic_census
        self.wiener_index = wiener_index
        self.node_connectivity = node_connectivity
        self.max_clique = max_clique
        self.average_clustering = average_clustering
        self.metric_closure = metric_closure
        self.ramsey_number = ramsey_number
        self.min_edge_dominating_set = min_egde_dominating_set
        self.degree_assortativity_coefficient = degree_assortativity_coefficient
        self.degree_pearson_correlation_coefficient = degree_pearson_correlation_coefficient
        self.average_degree_connectivity = average_degree_connectivity
        self.k_nearest_neighbors = k_nearest_neighbors
        self.graph_clique_number = graph_clique_number
        self.transivity =transivity
        self.communicability = communicability
        self.number_connected_components = number_connected_components
        self.number_strongly_connected_components = number_strongly_connected_components
        self.number_weakly_connected_components = number_weakly_connected_components
        self.number_attracting_components = number_attracting_components
        self.overall_reciprocity = overall_reciprocity
        self.rich_club_coefficient = rich_club_coefficient
        self.average_shortest_path_length = average_shortest_path_length
        self.current_flow_closeness_centrality = current_flow_closeness_centrality
        self.information_centrality = information_centrality
        self.betweeness_centrality = betweeness_centrality
        self.k_nodes_betweenness_centrality = k_nodes_betweenness_centrality
        self.edge_betweenness_centrality = edge_betweenness_centrality
        self.k_nodes_edge_betweenness_centrality = k_nodes_edge_betweenness_centrality
        self.communicability_betweenness_centrality = communicability_betweenness_centrality
        self.load_centrality = load_centrality
        self.edge_load_centrality = edge_load_centrality
        self.all_pairs_node_connectivity = all_pairs_node_connectivity
        self.all_pairs_node_connectivity_nbunch = all_pairs_node_connectivity_nbunch
        self.all_pairs_node_connectivity_cutoff = all_pairs_node_connectivity_cutoff

        self.average_neighbor_degree = average_neighbor_degree

        self.degree_centrality = degree_centrality
        self.eigenvector_centrality = eigenvector_centrality
        self.katz_centrality = katz_centrality
        self.closeness_centrality = closeness_centrality

        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        X_transformed = []

        #iterate over each individual graph and compute measures
        for i in X:

            print("computing graph stuff")
            A = global_efficiency(i)
            print(A)

            measure_list = []

            for m in self.measures:
                measure_list.append(measures_dict[m](i))
            if self.global_efficiency == 1:
                measure_list.append(global_efficiency(i))
            if self.global_reaching_centrality == 1:
                measure_list.append(global_reaching_centrality(i))
            if self.sigma == 1:
                measure_list.append(sigma(i))
            if self.omega == 1:
                measure_list.append(omega(i))
            if self.estrada_index == 1:
                measure_list.append(networkx.algorithms.centrality.estrada_index(i))
            if self.s_metric == 1:
                measure_list.append(networkx.algorithms.s_metric(i))
            if self.triadic_census == 1:
                measure_list.append(networkx.algorithms.triadic_census(i))
            if self.wiener_index == 1:
                measure_list.append(wiener_index(i))

            #approximation
            if self.node_connectivity == 1:
                measure_list.append(approximation.node_connectivity(i))
            if self.max_clique == 1:
                measure_list.append(approximation.max_clique(i))
            if self.average_clustering == 1:
                measure_list.append(approximation.average_clustering(i))
            if self.ramsey_number == 1:
                measure_list.append(approximation.ramsey_R2(i))

            #TODO: This returns a graph
            if self.metric_closure == 1:
                measure_list.append(approximation.steinertree.metric_closure(i))

            #Assortativity
            if self.degree_assortativity_coefficient == 1:
                measure_list.append(networkx.algorithms.degree_assortativity_coefficient(i))
            if self.degree_pearson_correlation_coefficient == 1:
                measure_list.append(networkx.degree_pearson_correlation_coefficient(i))
            #calculates average degree_connectivity for each node
            if self.average_degree_connectivity == 1:
                average_degree_conn = average_degree_connectivity(i)
                for key, val in average_degree_conn:
                    measure_list.append(val)
            if self.k_nearest_neighbors == 1:
                measure_list.append(networkx.k_nearest_neighbors(i))
            if self.graph_clique_number == 1:
                cliques = networkx.enumerate_all_cliques(i)
                measure_list.append(networkx.graph_clique_number(i, cliques))

            #transívity function
            if self.transivity == 1:
                measure_list.append(networkx.transitivity(i))

            #communicability
            if self.communicability == 1:
                measure_list.append(networkx.communicability(i))

            #components
            if self.number_connected_components == 1:
                measure_list.append(networkx.number_connected_components(i))
            if self.number_strongly_connected_components == 1:
                measure_list.append(networkx.number_strongly_connected_components(i))
            if self.number_weakly_connected_components == 1:
                measure_list.append(networkx.number_weakly_connected_components(i))
            if self.number_attracting_components == 1:
                measure_list.append(networkx.number_attracting_components(i))

            #Reciprocity
            if self.overall_reciprocity == 1:
                measure_list.append(networkx.overall_reciprocity(i))

            #Rich Club
            if self.rich_club_coefficient == 1:
                measure_list.append(networkx.rich_club_coefficient(i))

            #shortest path
            if self.average_shortest_path_length == 1:
                measure_list.append(networkx.average_shortest_path_length(i))

            if self.all_pairs_node_connectivity == 1:
                measure_list.append(approximation.all_pairs_node_connectivity(i, nbunch= self.all_pairs_node_connectivity_nbunch,
                                                                           cutoff= self.all_pairs_node_connectivity_cutoff))

            if self.average_neighbor_degree == 1:
                average_neighbour_values = average_neighbor_degree(i)
                #take dictionary and extract the values then append them to the list
                for key, val in average_neighbour_values:
                    measure_list.append(val)

            if self.degree_centrality == 1:
                degree_centrality_values = degree_centrality(i)
                for key, val in degree_centrality_values:
                    measure_list.append(val)

            if self.eigenvector_centrality == 1:
                eigenvector_centrality_values = eigenvector_centrality(i)
                for key, val in eigenvector_centrality_values:
                    measure_list.append(val)

            if self.katz_centrality == 1:
                katz_centrality_values = katz_centrality(i)
                for key, val in katz_centrality_values:
                    measure_list.append(val)

            if self.closeness_centrality == 1:
                closeness_centrality_values = closeness_centrality(i)
                for key, val in closeness_centrality_values:
                    measure_list.append(val)

            if self.current_flow_closeness_centrality == 1:
                current_flow_closeness_centrality_values = current_flow_closeness_centrality(i)
                for key, val in current_flow_closeness_centrality_values:
                    measure_list.append(val)

            if self.information_centrality == 1:
                information_centrality_values = information_centrality(i)
                for key, val in information_centrality_values:
                    measure_list.append(val)

            if self.betweeness_centrality == 1:
                betweeness_centrality_values = betweenness_centrality(i, k=self.k_nodes_betweenness_centrality)
                for key, val in betweeness_centrality_values:
                    measure_list.append(val)

            if self.edge_betweenness_centrality == 1:
                edge_betweenness_centrality_values = edge_betweenness_centrality(i, k=self.k_nodes_edge_betweenness_centrality)
                for key, val in edge_betweenness_centrality_values:
                    measure_list.append(val)

            if self.communicability_betweenness_centrality == 1:
                communicability_betweenness_centrality_values = communicability_betweenness_centrality(i)
                for key, val in communicability_betweenness_centrality_values:
                    measure_list.append(val)

            if self.load_centrality == 1:
                load_centrality_values = load_centrality(i)
                for key, val in load_centrality_values:
                    measure_list.append(val)

            if self.edge_load_centrality == 1:
                edge_load_centrality_values = edge_load_centrality(i)
                for key, val in edge_load_centrality_values:
                    measure_list.append(val)

            X_transformed.append(measure_list)

        np.asarray(X_transformed)

        return X_transformed


class LocalGraphMeasureTransform(BaseEstimator, TransformerMixin):
    _estimator_type = "transformer"

    def __init__(self,
                 all_pairs_node_connectivity = 0, all_pairs_node_connectivity_nbunch = None,
                 all_pairs_node_connectivity_cutoff = None,
                 average_neighbor_degree = 0, degree_centrality = 0,
                 eigenvector_centrality = 0, katz_centrality = 0,
                 closeness_centrality = 0,
                 logs=''):

        #self.
        if logs:
            self.logs = logs
        else:
            self.logs = os.getcwd()

    def fit(self, X, y):
        pass

    def transform(self, X):

        X_transformed = []

        #computes the node connectivity between all pairs of nodes
        for i in X:

            measure_list = []






            X_transformed.append(measure_list)


        #if self.

        np.asarray(X_transformed)

        return X_transformed
