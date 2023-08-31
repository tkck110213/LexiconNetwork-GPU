# Import needed libraries
import cugraph
import cudf
from cugraph.experimental import PropertyGraph
import cupy as cp
from gensim.models import KeyedVectors
import json
import sys
import pickle
import os
import Levenshtein
import itertools
import pyopenjtalk



class LexiconNetwork:
    def __init__(self, vectorPath):
        print(f"model path:{vectorPath}")
        print("[\033[2m+\033[m] Loading word vector model...")
        self.model = KeyedVectors.load_word2vec_format(vectorPath)
        print("[\033[2m+\033[m]\033[1;32m Sucsess to load word vectors !! \033[m")

        self.wordlist = [word.replace("#", "") for word in self.model.index_to_key]
        self.dimention = len(self.wordlist)
        self.wordVectors = cp.array([self.model.get_vector(word) for word in self.model.index_to_key])
        self.SemanticNetwork = PropertyGraph()
        self.PhonologicalNetwork = PropertyGraph()

        self.makePhonologicalNetwork()
        
    def makeSemanticNetwork(self):
        """Add vertex(word) in lexicon netowrk graph"""
        print("[\033[2m+\033[m] Add word in lexicon network...")
        vertDf = cudf.DataFrame({"id":[i for i in range(self.dimention)], "label":self.wordlist, "reservior":cp.zeros(self.dimention), 
                                 "inflow":cp.zeros(self.dimention), "outflow":cp.zeros(self.dimention)})
        self.SemanticNetwork.add_vertex_data(vertDf, vertex_col_name="id")
        
        """Calc cosine similarity of each words"""
        print("[\033[2m+\033[m] Calc similarity of each words...")
        # calc norm vector following row 
        normVectors = cp.linalg.norm(self.wordVectors, axis=1)
        similarities = cp.dot(self.wordVectors, self.wordVectors.T) / cp.dot(normVectors, normVectors.T)
        # replace similarity of one of duplication edge pair(1-2:2-1) to cp.nan
        similarities[cp.tril_indices(similarities.shape[0])] = cp.nan
        similarities = cp.ravel(similarities)
        

        """Add edge to lexicon netowrk graph"""
        print("[\033[2m+\033[m] Add edge in semantic network...")
        # make combination edges pair
        vertexList = cp.array(range(self.dimention))
        edgeList = cp.array(cp.meshgrid(vertexList, vertexList)).T.reshape(-1, 2)
        # make edge dataframe
        edgeDf = cudf.DataFrame({})
        edgeDf["src"] = edgeList[:, 0]
        edgeDf["dst"] = edgeList[:, 1]
        edgeDf["weight"] = similarities
        # delete one of duplication edge pair
        edgeDf = edgeDf.dropna(how="any")
        self.SemanticNetwork.add_edge_data(edgeDf, vertex_col_names=("src", "dst"))
        print("[\033[2m+\033[m]\033[1;32m Sucsess to generate semantic network !! \033[m")
        print(self.SemanticNetwork.get_vertex_data())
        print(self.SemanticNetwork.get_edge_data())



    def makePhonologicalNetwork(self):
        print("[\033[2m+\033[m] Add word in phonological network...")
        vertDf = cudf.DataFrame({"id":[i for i in range(self.dimention)], "label":self.wordlist, "reservior":cp.zeros(self.dimention), 
                                 "inflow":cp.zeros(self.dimention), "outflow":cp.zeros(self.dimention)})
        self.PhonologicalNetwork.add_vertex_data(vertDf, vertex_col_name="id")

        # make edge dataframe
        edgeDf = cudf.DataFrame({})
        wordIdPerm = list(itertools.permutations(range(self.dimention), 2))
        print(len([Id[0] for Id in wordIdPerm]))
        print(len([Id[1] for Id in wordIdPerm]))
        edgeDf["src"] = [Id[0] for Id in wordIdPerm]
        edgeDf["dst"] = [Id[1] for Id in wordIdPerm]
        edgeDf["distance"] = [Levenshtein.distance(pyopenjtalk.g2p(self.wordlist[Id[0]]), pyopenjtalk.g2p(self.wordlist[Id[1]])) for Id in wordIdPerm]
        edgeDf = edgeDf[edgeDf.distance < 2]
        self.PhonologicalNetwork.add_edge_data(edgeDf, vertex_col_names=("src", "dst"))
        print(edgeDf)
        #print(self.PhonologicalNetwork.get_vertex_data())
        #print(self.PhonologicalNetwork.get_edge_data())

    def saveLexiconNetwork(self, savePath):
        """Export graph object (pickle)"""
        with open(savePath, "wb") as f:
            pickle.dump(self.SemanticNetwork, f)
            os.chmod(savePath, 755)
            print(f"[\033[2m+\033[m] Save to {savePath}")
        

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        setting = json.load(f)

    sn = LexiconNetwork(setting["vector_path"])
    sn.makePhonologicalNetwork()
    #sn.saveLexiconNetwork(setting["save_path"])

