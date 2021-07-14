import numpy as np
import json
import heapq
import time


class Pathfinder:
    def __init__(self, graph_path, a_star=False):
        with open(graph_path, "r") as f:
            self.graph = json.load(f)
        self.use_aStar = a_star
        self.nodes = self._get_node_attributes()

    def dijkstra(self, source, target):
        pq = []
        distTo = {}
        edgeTo = {}
        seen = {k: False for k in self.nodes.keys()}

        source = self.nodes[source]
        target = self.nodes[target]

        heapq.heappush(pq, (0, source.name))
        distTo[eval(source.name)] = 0
        seen[eval(source.name)] = True
        for n in self.nodes.keys():
            if n != eval(source.name):
                distTo[n] = np.inf
        while len(pq) > 0:
            # print("Distances: ", distTo)
            p = heapq.heappop(pq)
            p = self.nodes[int(p[1])]
            for n in p.neighbors:
                nm = n[0]
                dist = n[1]
                if not seen[nm]:
                    heapq.heappush(pq, (dist, str(nm)))
                    seen[nm] = True
            pq, edgeTo, distTo = self.relax(p, distTo, edgeTo, pq)

        try:
            k = target.name
            path = []
            while int(k) != int(source.name):
                path.append(k)
                k = edgeTo[int(k)]
            path.append(source.name)
            # print("Path Taken: ", path[::-1])
        except KeyError:
            pass
            # print("No path found between the provided points.")
        return distTo[eval(target.name)]

    def relax(self, node, distTo, edgeTo, pq):
        name = eval(node.name)
        for q in node.neighbors:
            qName = q[0]
            qWeight = q[1]
            if not self.use_aStar:
                if distTo[name] + qWeight < distTo[qName]:
                    removal = [x for x in pq if x[1] == str(qName)]
                    if removal:
                        pq.remove(removal[0])
                    distTo[qName] = distTo[name] + qWeight
                    edgeTo[qName] = node.name
                    heapq.heappush(pq, (distTo[qName], str(qName)))
            else:
                qNode = self.nodes[qName]
                if distTo[name] + qWeight + euclidean_distance(qNode.position, node.position) < distTo[qName]:
                    removal = [x for x in pq if x[1] == str(qName)]
                    if len(removal) > 0:
                        pq.remove(removal[0])
                    distTo[qName] = distTo[name] + qWeight
                    edgeTo[qName] = node.name
                    heapq.heappush(pq, (distTo[qName], str(qName)))
                    heapq.heapify(pq)
        return pq, edgeTo, distTo

    def _get_node_attributes(self):
        names = [k for k in list(self.graph.keys())]
        nodes = {}
        for name in names:
            nd = self.graph[name][0]
            nodes[eval(name)] = (Node(name, nd))
        return nodes


class Node:
    def __init__(self, name, params):
        self.name = name
        self.position = params['position']
        self.neighbors = params['neighbors']

    def __repr__(self):
        return f"[Name: {self.name}, Neighbors: {self.neighbors}]"


def nearest(point, nodes):
    out = -1
    min_distance = np.inf
    for node in nodes.values():
        pos = node.position
        ed = euclidean_distance(pos, point)
        if ed < min_distance:
            min_distance = ed
            out = int(node.name)
    return out


def euclidean_distance(pt1, pt2):
    return np.sqrt(((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2))


def main():
    pf = Pathfinder("graphs/idpSanteJuanApt.json", a_star=False)

    t1 = time.time()

    pt1 = (5.047613999999999, 1.2285701999999998)
    pt2 = (1.2761892, 1.0380942)
    print("Straight-line distance: ", euclidean_distance(pt1, pt2))

    a = nearest(pt1, pf.nodes)
    b = nearest(pt2, pf.nodes)
    print(a, b)
    print(pf.nodes[a].position, pf.nodes[b].position)
    d = pf.dijkstra(a, b)
    t2 = time.time()
    print("Shortest Path Length: ", d)
    print(f"completed in {t2 - t1} seconds")


if __name__ == '__main__':
    main()
