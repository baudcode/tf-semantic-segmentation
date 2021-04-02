from enum import unique
from PIL import Image
from IPython.display import display
import cv2
import networkx as nx
from typing import List, Tuple
from shapely.geometry import Polygon
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, Point, GeometryCollection, Polygon
from functools import reduce
from tf_semantic_segmentation.visualizations.masks import get_rgb
import copy
from tf_semantic_segmentation.utils import logger


print = logger.debug


def get_contours_min_max(contours):
    xs = [cnt[:, 0].tolist() for cnt in contours]
    xs = list(reduce(lambda x, y: x + y, xs))
    xs = np.asarray(xs)

    ys = [cnt[:, 1].tolist() for cnt in contours]
    ys = list(reduce(lambda x, y: x + y, ys))
    ys = np.asarray(ys)

    xmax = xs.max()
    ymax = ys.max()

    xmin = xs.min()
    ymin = ys.min()
    return [xmin, ymin, xmax, ymax]


def azimuth(point1: Point, point2: Point) -> float:
    '''azimuth between 2 shapely points (interval 0 - 360)'''
    angle = np.arctan2(point2.y - point1.y, point2.x - point1.x)
    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360


def get_start_end(ls: LineString) -> Tuple[np.ndarray, np.ndarray]:
    x, y = ls.coords.xy
    line = np.asarray(list(zip(x, y)))
    return line[0], line[1]


def remove_duplicates(outline):
    outline = np.asarray(outline)
    _, index = np.unique(outline, axis=0, return_index=True)

    # return unique entries in the correct order
    index.sort()
    outline = outline[index]
    return outline


def poly2arr(g: Polygon) -> np.ndarray:
    if g.is_empty:
        return []

    x = g.exterior.coords.xy[0]
    y = g.exterior.coords.xy[1]
    return np.asarray(list(zip(x, y)))


def _line_merger(lines):
    found = []
    clusters = []

    lines = sorted(lines, key=lambda x: -x.length)
    for i, l in enumerate(lines):
        if i in found:
            continue

        cluster = [i]
        for i2 in range(i + 1, len(lines)):
            if i2 in found:
                continue

            l2 = lines[i2]

            if l.intersects(l2):
                found.append(i2)
                cluster.append(i2)

        clusters.append(cluster)

    for cluster in clusters:
        cluster_lines = [lines[i] for i in cluster]

        points = []

        for ls in cluster_lines:
            x, y = ls.coords.xy
            points.extend(list(map(list, zip(x, y))))

        minmax = get_contours_min_max([np.asarray(points)])
        # sort by distance to 0, 0
        points = sorted(points, key=lambda p: np.linalg.norm(np.asarray(p) - np.asarray(minmax[:2])))
        yield LineString([points[0], points[-1]])


def line_merger(lines, iterations: int = 5):
    for _ in range(iterations):
        num = len(lines)
        lines = list(_line_merger(lines))

        if len(lines) == num:
            # pre exit when the number of lines did not change
            break

    return lines


def round_geometry(g, precision: int = 2):
    from shapely.geometry import mapping, shape
    m = mapping(g)

    result = []
    for i, coord in enumerate(m['coordinates']):
        result.append([round(coord[0], precision), round(coord[1], precision)])
    m['coordinates'] = result
    return shape(m)


def most_common(lst):
    return max(set(lst), key=lst.count)


def find_longest_path(g: nx.Graph):
    u = list(g.nodes)[0]
    v = list(g[u])[0]

    return max(nx.all_simple_paths(g, u, v), key=lambda x: len(x))


def get_edges_from_list(l: List[str], roundtrip: bool = False) -> List[Tuple[str, str]]:
    edges = []
    for i, c in enumerate(l):
        if i == (len(l) - 1) and not roundtrip:
            continue
        edges.append((c, l[(i + 1) % len(l)]))
    return edges


def get_contour(g: nx.Graph):
    path = find_longest_path(g)
    print("path: ", path)
    points = [g.nodes[node]['point'] for node in path]
    return np.asarray(points)


def get_line(g: nx.Graph, edge):
    p1 = g.nodes[edge[0]]['point']
    p2 = g.nodes[edge[1]]['point']
    return LineString([p1, p2])


def get_graph_from_lines(lines):
    g = nx.Graph()

    for i, l in enumerate(lines):
        points = np.asarray(l).reshape((-1, 2))
        u = "line_%d_0" % i
        v = "line_%d_1" % i
        print(points[0], points[1])
        g.add_node(u, point=points[0])
        g.add_node(v, point=points[1])
        g.add_edge(u, v)

    # find closest edges
    nodes = list(g.nodes)
    for node in nodes:
        point = g.nodes[node]['point']
        edges = g.edges([node])

        if len(edges) == 0:
            print("FAIL")
        if len(edges) == 2:
            continue

        closest_node = None
        closest_distance = 1e10

        for node2 in nodes:
            if node != node2:
                distance = np.linalg.norm(g.nodes[node2]['point'] - point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_node = node2
        if closest_node == None:
            print("FAIL")

        g.add_edge(node, closest_node)
    return g


def get_points_and_nodes(g: nx.Graph):
    nodes = list(g.nodes)
    return np.asarray([g.nodes[n]['point'] for n in nodes]), np.asarray(nodes)


def get_start_end_from_points(points, line):
    start, _ = get_start_end(line)
    # sort by distance to 0, 0
    points = sorted(points, key=lambda p: np.linalg.norm(np.asarray(p) - start))
    return (points[0], points[-1])


def cluster_close_points(g: nx.Graph, eps: float = 15):
    X, nodes = get_points_and_nodes(g)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(X)

    unique_labels = list(set(clustering.labels_.tolist()) - set([-1]))
    unique_labels = sorted(unique_labels, key=lambda label: len(np.where(clustering.labels_ == label)), reverse=True)

    for label in unique_labels:
        idxs = np.where(clustering.labels_ == label)
        points = X[idxs]
        cluster_nodes = nodes[idxs]

        if len(points) < 2:
            continue

        centroid = np.mean(points, axis=0)
        print("Reduce points", points)
        print("Reduce nodes", cluster_nodes, "to centroid", centroid)
        connected = set([v for node in cluster_nodes for v in g[node]])
        connected = connected - set(cluster_nodes)

        for node in cluster_nodes:
            g.remove_node(node)

        cluster_node = "cluster_%d" % label
        g.add_node(cluster_node, point=centroid)

        for node in connected:
            g.add_edge(cluster_node, node)

    return g


def line_iterator(contour):
    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]
        yield LineString([p1, p2])


def line_iterator_arr(contour):
    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]
        yield np.asarray([p1, p2])


def to_tuple(p):
    return tuple(list(map(int, p)))


def get_intersection(p, d, bounds, step_size, max_steps):
    for i in range(1, max_steps + 1):
        p2 = p + d * (step_size * i)
        line = LineString([p, p2])
        if line.intersects(bounds):
            return np.asarray(line.intersection(bounds))

    raise Exception("cannot find intersection")


def _create_channels(g: nx.Graph, buffer, bounds, step_size, max_steps):
    lines = []
    edges = list(g.edges)
    extended_lines = []

    for u, v in edges:
        uc = g.nodes[u]['point']
        uv = g.nodes[v]['point']

        length = np.linalg.norm(uc - uv)
        d = (uc - uv) / length
        extended_uc = get_intersection(uc, d, bounds, step_size, max_steps)
        extended_uv = get_intersection(uv, -d, bounds, step_size, max_steps)

        extended_lines.append(LineString([extended_uc, extended_uv]))
        lines.append(LineString([uc, uv]))

    line_clusters = []
    channels = []

    found = []
    # find lines that intersect
    for i, l in enumerate(extended_lines):
        cluster = [i]

        if i in found:
            continue

        for j in range(i + 1, len(lines)):
            if j in found:
                continue

            if l.buffer(buffer).contains(lines[j]):
                # print("match:")
                # display(GeometryCollection([l.buffer(buffer), lines[j]]))
                cluster.append(j)
                found.append(j)

        found.append(i)
        line_clusters.append(cluster)
        channels.append(l)

    # find start and end inside channel
    cluster_lengths = []
    cluster_start_end = []
    line_clusters = [(c, channels[i]) for i, c in enumerate(line_clusters) if len(c) > 1]
    clusters = []

    for cluster, channel in line_clusters:
        unique_nodes = list(set([node for i in cluster for node in edges[i]]))
        for nodes in separate_nodes_by_cc(g, unique_nodes):
            valid_idx = [i for i in cluster if any(node in edges[i] for node in nodes)]
            points = [point for i in valid_idx for point in get_start_end(lines[i])]
            start, end = get_start_end_from_points(points, channel)
            length = LineString([start, end]).length
            cluster_lengths.append(length)
            cluster_start_end.append((start, end))
            clusters.append((valid_idx, channel))

    return lines, edges, line_clusters, cluster_lengths, cluster_start_end


def get_closest(points, dst):
    lenghts = [np.linalg.norm(p - dst) for p in points]
    return np.argmin(lenghts)


def find_unconnected(g: nx.Graph, sorted_nodes):

    for node in sorted_nodes:
        other_edges = list(filter(lambda x: x not in sorted_nodes, g[node]))
        if len(other_edges) > 0:
            return other_edges[0]

    return None


def separate_nodes_by_cc(g, nodes):
    g2 = nx.Graph()
    for node in nodes:
        g2.add_node(node)
        for edge in g.edges(node):
            g2.add_edge(*edge)

    return list(nx.connected_components(g2))


def create_channels(g: nx.Graph, size, step_size: int = 5, buffer: int = 15):
    diag = np.sqrt(pow(size[0], 2) + pow(size[1], 2))
    max_steps = int(diag // step_size)
    bounds = np.asarray([
        [0, 0],
        [size[0], 0],
        [size[0], size[1]],
        [0, size[1]]
    ])
    bounds = Polygon(bounds).boundary

    has_channels = True

    while has_channels:
        lines, edges, clusters, cluster_lengths, cluster_start_end = _create_channels(g, buffer, bounds, step_size, max_steps)
        print("found %d clusters" % len(clusters))
        if len(clusters) == 0:
            has_channels = False
        else:
            argmax = np.argmax(cluster_lengths)
            idxs = clusters[argmax][0]

            start, end = cluster_start_end[argmax]
            cluster_edges = [edges[idx] for idx in idxs]
            cluster_lines = [lines[idx] for idx in idxs]

            unique_nodes = list(set([node for edge in cluster_edges for node in edge]))
            sorted_nodes = sorted(unique_nodes, key=lambda x: np.linalg.norm(g.nodes[x]['point'] - start))
            points = [g.nodes[n]['point'] for n in sorted_nodes]
            print("points: ", np.asarray(points).tolist())
            print("lines: ", [str(l) for l in cluster_lines])

            start_v = find_unconnected(g, sorted_nodes)
            end_v = find_unconnected(g, list(reversed(sorted_nodes)))

            start_idx = get_closest(points, start)
            end_idx = get_closest(points, end)

            start_node = sorted_nodes[start_idx]
            end_node = sorted_nodes[end_idx]

            print("from ", start_node, "to", end_node)

            print("deleting nodes", sorted_nodes)
            for node in sorted_nodes:
                g.remove_node(node)

            print("adding node", start_node, "at", start)
            print("adding node", end_node, "at", end)
            g.add_node(start_node, point=start)
            g.add_node(end_node, point=end)
            g.add_edges_from([(start_node, end_node), (end_node, end_v), (start_node, start_v)])
        # sort by len of lines

    # print("found clusters: ", line_clusters)

    # node_to_point = {n: g.nodes[n]['point'] for n in g.nodes}
    # original_edges = {n: [v for v in g[n]] for n in g.nodes}
    # # start, end, point

    # for idxs in line_clusters:
    #     _lines = [lines[idx] for idx in idxs]
    #     if len(_lines) <= 1:
    #         continue

    #     print("lines: ", _lines)
    #     cluster_edges = [edges[idx] for idx in idxs]
    #     unique_lines = list(line_merger(_lines))

    #     print("found %d unique lines" % len(unique_lines))
    #     display(GeometryCollection(unique_lines))

    #     for li, ls in enumerate(unique_lines):
    #         start_point, end_point = get_start_end(ls)
    #         print("[LINE STATUS] line %d from %s to %s" % (li, start_point, end_point))

    #         # sort all points on the same line according to distance
    #         valid_edges = {edge: lines[i] for i, edge in enumerate(cluster_edges) if _lines[i].intersects(ls)}
    #         print("valid edges length=%d keys=%s" % (len(valid_edges), str(valid_edges.keys())))

    #         unique_nodes = list(set([node for edge in valid_edges for node in edge]))
    #         print("unique nodes: %s" % str(unique_nodes))

    #         # sort corners by distance to start_point
    #         sorted_nodes = sorted(unique_nodes, key=lambda x: np.linalg.norm(node_to_point[x] - start_point))
    #         print("sorted (keeping first and last): ", sorted_nodes)

    #         start = sorted_nodes[0]
    #         end = sorted_nodes[-1]

    #         # find other connection from start to end
    #         start_v, end_v = None, None

    #         for node in sorted_nodes:
    #             other_edges = list(filter(lambda x: x not in unique_nodes, g[node] if node in g else original_edges[node]))
    #             if len(other_edges) > 0:
    #                 start_v = other_edges[0]
    #                 break

    #         for node in reversed(sorted_nodes):
    #             other_edges = list(filter(lambda x: x not in unique_nodes, g[node]))
    #             if len(other_edges) > 0:
    #                 end_v = other_edges[0]
    #                 break
    #         # remove all other edges

    #         print("start/end v: ", start_v, end_v)
    #         intermediate_nodes = sorted_nodes[1:-1]

    #         for node in sorted_nodes:
    #             sorted_node_edges = list(g.edges(node))

    #             for edge in sorted_node_edges:
    #                 print("removing intermediate edge in cluster", edge)
    #                 g.remove_edge(*edge)

    #         # remove intermediate nodes
    #         for node in intermediate_nodes:
    #             print("remove intermediate node", node)
    #             if node in g:
    #                 g.remove_node(node)

    #         print("adding edge from start to end of line cluster", start, end)
    #         g.add_edge(start, end)
    #         g.add_edge(start, start_v)
    #         g.add_edge(end, end_v)

    #     nodes = list(g.nodes)

    #     for node in nodes:
    #         if len(g[node]) == 0:
    #             print("removing node", node)
    #             g.remove_node(node)

    return g


def draw_graph(mask: np.ndarray, g: nx.Graph, radius: int = 5, point_color: tuple = (255, 0, 0), line_color: tuple = (0, 255, 0), line_thickness: int = 2, show: bool = False):
    contour = get_contour(g)
    image = get_mask_rgb(mask)

    for line in line_iterator_arr(contour):
        start, end = line
        # draw line
        print(to_tuple(start), to_tuple(end), image.shape, image.dtype)
        cv2.line(image, to_tuple(start), to_tuple(end), line_color, thickness=line_thickness)

    for p in contour:
        image = cv2.circle(image, to_tuple(p), radius, point_color, thickness=cv2.FILLED)

    if show:
        display(Image.fromarray(image))

    return image


def process_batch(masks, draw_on, buffer: int = 15, eps: int = 10, binary_threshold: float = 0.5):
    detector = cv2.ximgproc.createFastLineDetector()
    processed = []
    for i, mask in enumerate(masks):
        mask = np.argmax(mask, axis=-1)
        mask[mask >= binary_threshold] = 255
        mask[mask < binary_threshold] = 0
        mask = mask.astype(np.uint8)
        lines = detector.detect(mask)
        image = get_rgb(draw_on[i].copy())
        try:
            g = get_graph_from_lines(lines)
            try:
                g = cluster_close_points(g, eps=eps)
                g = create_channels(copy.deepcopy(g), mask.shape[:2][::-1], buffer=buffer)
            except:
                pass
            image = draw_graph(image, g)

        except Exception as e:
            print("error: %s" % str(e))
            center = (image.shape[1] // 2, image.shape[0] // 2)
            cv2.putText(image, "Error", center, cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), thickness=5)

        print(image.max(), image.min(), image.dtype, image.shape)
        processed.append(image)
    return processed
