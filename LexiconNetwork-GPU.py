# Import needed libraries
import time
import cugraph
import cudf
from cugraph.experimental import PropertyGraph

import cupy as cp

import networkx as nx
from gensim.models import KeyedVectors
import re
from tqdm import tqdm
import sys
import pickle
import os

class SimilarNetwork:
    def __init__(self):
        path = f'/mnt/Project/Resource/jawiki.all_vectors.200d.20000w.stop.txt'
        print(f"model path:{path}")
        print("Word2Vec loading...")
        self.model = KeyedVectors.load_word2vec_format(path)
        print("Complete Word2Vec loading")

        self.wordlist = self.model.index_to_key
        self.dimention = len(self.wordlist)
        self.wordVectors = cp.array([self.model.get_vector(word) for word in self.model.index_to_key])
        self.similarityNetwork = PropertyGraph()
        
    def makeSimilarNetwork(self):
        vertDf = cudf.DataFrame({"id":[i for i in range(len(self.wordlist))], "label":self.wordlist})
        #print(vertDf)
        
        self.similarityNetwork.add_vertex_data(vertDf, vertex_col_name="id")
        """
        for i in tqdm(range(len(self.wordlist))):
            for j in range(1, len(self.wordlist)):
                similarity = self.model.similarity(self.wordlist[i], self.wordlist[j])
                edgeDf = cudf.DataFrame(columns=["src", "dst", "weight"], data=[(i, j, similarity)])
                self.similarityNetwork.add_edge_data(edgeDf, vertex_col_names=("src", "dst"))
        """
        """
        for v_i, v in enumerate(tqdm(self.wordVectors)):
            #print(v_i, range(v_i + 1, len(self.wordVectors)))
            similarities = cp.dot(self.wordVectors[v_i + 1:], v.T) / (cp.linalg.norm(v) * cp.linalg.norm(self.wordVectors[v_i + 1:], axis=1))
            edgeDf = cudf.DataFrame(columns=["src", "dst", "weight"], data=[(v_i, v_j, similarity) for v_j, similarity in zip(range(v_i + 1, len(self.wordVectors)), similarities)])
            self.similarityNetwork.add_edge_data(edgeDf, vertex_col_names=("src", "dst"))
        """
        normVectors = cp.linalg.norm(self.wordVectors, axis=1)
        similarities = cp.dot(self.wordVectors, self.wordVectors.T) / cp.dot(normVectors, normVectors.T)
        #print(similarities)
        similarities[cp.tril_indices(similarities.shape[0])] = cp.nan
        #print(similarities)
        similarities = cp.ravel(similarities)
        #print(similarities)
        #print(similarities.shape)
        vertexList = cp.array(range(self.dimention))
        edgeList = cp.array(cp.meshgrid(vertexList, vertexList)).T.reshape(-1, 2)
        #print(edgeList)
        #print(edgeList.shape)
        edgeDf = cudf.DataFrame({})
        edgeDf["src"] = edgeList[:, 0]
        edgeDf["dst"] = edgeList[:, 1]
        #print(edgeDf)
        edgeDf["weight"] = similarities
        edgeDf = edgeDf.dropna(how="any")
        #print(edgeDf.info)
        #print(normVectors.shape)
        #print(similarities.shape)
        #edgeDf = cudf.DataFrame(columns=["src", "dst", "weight"], data=[(i, j, similarities[i, j]) for i in tqdm(range(self.dimention - 1)) for j in range(i + 1, self.dimention)])
        self.similarityNetwork.add_edge_data(edgeDf, vertex_col_names=("src", "dst"))

        with open(f'/mnt/Project/Resource/SimilarityNetwor_test_complete_20000w_2.pkl', "wb") as f:
            pickle.dump(self.similarityNetwork, f)
            os.chmod(f'/mnt/Project/Resource/SimilarityNetwor_test_complete_20000w_2.pkl', 755)
        


def main():
    sn = SimilarNetwork()
    sn.makeSimilarNetwork()

main()
