from g_basic import Client
from g_message import *

if __name__ == "__main__":
    client = Client(address="tcp://localhost:5555")
    randompoint = np.random.rand(1000, 3) * 0.5 + 0.1
    randompoint1 = np.random.rand(1000, 3) * 0.5 + 1.0
    vertice = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2]])
    test_pt = GMesPointCloud.from_numpy((randompoint, 0))
    test_pt1 = GMesPointCloud.from_numpy((randompoint1, 1))
    test_mesh = GMesTrimesh.from_numpy((vertice, faces, 0))
    test_whole = GMesssage(pointcloud=[test_pt, test_pt1], trimesh=[test_mesh])
    client.send(GMesssage.serialize(test_whole))
    client.close()