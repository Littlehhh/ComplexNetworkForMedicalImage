import os
import numpy as np
import matplotlib.pyplot as plt
import  networkx as nx
import pandas as pd
import seaborn as sns
from DataPatch import compute_all_brain,compute_all_hippocampus

if __name__ == '__main__':
    if not os.path.exists("ave_patches.npy"):
        compute_all_brain('D:\\Data\\Registration_Philip3')
    if not os.path.exists("hipp_label_patches.npy"):
        compute_all_hippocampus('D:\\Data\\Registration_Philip3_Hippocampus')

    ave_data = np.load('ave_patches.npy')
    # matrix_r = np.corrcoef(ave_data)
    # matrix_r_abs = np.abs(matrix_r)
    hipp_data = np.load('hipp_label_patches.npy')
    data_list = np.where(np.sum(ave_data, axis=1) > 10000)
    hipp_list = np.where(np.sum(hipp_data, axis=1) > 0.1)
    print(np.sum(hipp_data))
    for seed in hipp_list[0]:
        if seed not in data_list[0]:
            print("Not all hipp in ave_list")
    end_nodes = ave_data[data_list]
    start_nodes = hipp_data[hipp_list]

    matrix_r = np.corrcoef(end_nodes)
    matrix_r_abs = np.abs(matrix_r)
    s = np.zeros(len(matrix_r_abs))
    Y = np.zeros(len(matrix_r_abs))
    for i in range(len(matrix_r_abs)):
        for j in range(len(matrix_r_abs)):
            if matrix_r_abs[i,j] > 0.3 and i != j:
                s[i] = s[i]+1
    for i in range(len(matrix_r_abs)):
        for j in range(len(matrix_r_abs)):
            if matrix_r_abs[i,j] > 0.3 and i != j:
                Y[i] = Y[i] + pow((matrix_r_abs[i,j]/s[i]), 2)
    plt.plot(Y)
    df = pd.DataFrame(matrix_r_abs)
    print(df)
    # sns.heatmap(df, annot=True)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix_r_abs)

    plt.show()

    # graph = nx.Graph()
    # for origin in hipp_list[0]:
    #     for end in data_list[0]:
    #         if origin != end:
    #             x = np.where(data_list == origin)
    #             graph.add_edge(origin,
    #                            end,
    #                            weight=matrix_r_abs[np.where(data_list == origin),
    #                                                np.where(data_list == end)])
    #
    # elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 0.97]
    # esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= 0.97]
    #
    # pos = nx.spring_layout(graph)  # positions for all nodes
    # # nodes
    # nx.draw_networkx_nodes(graph, pos, node_size=1.5)
    #
    # # edges
    # nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=0.8)
    # nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=0.8, alpha=0.5, edge_color='b', style='dashed')
    #
    # # labels
    # # plt.figure(figsize=(14, 10) )
    # nx.draw_networkx_labels(graph, pos, font_size=1, font_family='sans-serif')
    # plt.axis('off')