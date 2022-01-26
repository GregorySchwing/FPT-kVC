import cudf
import cugraph

# read data into a cuDF DataFrame using read_csv
gdf = cudf.read_csv("25_nodes.csv", names=["src", "dst"], dtype=["int32", "int32"])

# We now have data as edge pairs
# create a Graph using the source (src) and destination (dst) vertex pairs
G = cugraph.Graph()
G.from_cudf_edgelist(gdf, source='src', destination='dst')

df = cugraph.overlap(G)
