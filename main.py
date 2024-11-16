from edmonds_karp import edmonds_karp, grafo1
from prim import prim_mst
from tsp import tsp_ant_colony_optimization

from typing import List, Tuple

import numpy as np
import sys

def readFile(filename: str = "input.txt") -> Tuple[int, List[List[int]], List[List[int]]]:
    n: int = 0
    adj_matrix: List[List[int]] = []
    cap_matrix: List[List[int]] = []
    with open(filename, 'r') as file:
        n_str: str = file.readline().strip()
        n: int = int(n_str)

        # Reads an empty line that separates the adjacency matrix. 
        file.readline()

        for _ in range(n):
            row: str = file.readline().strip()
            row_strs: List[str] = row.split(" ")
            row_filtered: List[str] = list(filter(lambda x: x != "", row_strs))

            row_numbers: List[int] = [int(n) for n in row_filtered]
            adj_matrix.append(row_numbers)

        # Reads an empty line that separates the capacity matrix. 
        file.readline()

        for _ in range(n):
            row: str = file.readline().strip()
            row_strs: List[str] = row.split(" ")
            row_filtered: List[str] = list(filter(lambda x: x != "", row_strs))

            row_numbers: List[int] = [int(n) for n in row_filtered]
            cap_matrix.append(row_numbers)
    return n, adj_matrix, cap_matrix

if __name__ == "__main__":
    n, adj_matrix, cap_matrix = readFile(filename="input.txt")
    adj_matrix_np = np.asmatrix(adj_matrix)

    print(prim_mst(n, adj_matrix))
    print()
    print(tsp_ant_colony_optimization(adj_matrix_np))
    print()
    print(edmonds_karp(n, adj_matrix, cap_matrix))