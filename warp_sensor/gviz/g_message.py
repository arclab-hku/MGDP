import proto 
import numpy as np

# class GMesTime(proto.Message):
#     sec = proto.Field(proto.INT64, number=1)
#     nsec = proto.Field(proto.INT64, number=2)

# class GMesHeader(proto.Message):
#     cls = proto.Field(proto.STRING, number=1)
#     time = proto.Field(GMesTime, number=2)
class GMesPoint(proto.Message):
    x = proto.Field(proto.FLOAT, number=1)
    y = proto.Field(proto.FLOAT, number=2)
    z = proto.Field(proto.FLOAT, number=3)

class GMesFace(proto.Message):
    a = proto.Field(proto.INT32, number=1)
    b = proto.Field(proto.INT32, number=2)
    c = proto.Field(proto.INT32, number=3)

class GMesPoints(proto.Message):
    x = proto.RepeatedField(proto.FLOAT, number=1)
    y = proto.RepeatedField(proto.FLOAT, number=2)
    z = proto.RepeatedField(proto.FLOAT, number=3)

class GMesFaces(proto.Message):
    a = proto.RepeatedField(proto.INT32, number=1)
    b = proto.RepeatedField(proto.INT32, number=2)
    c = proto.RepeatedField(proto.INT32, number=3)

class GMesPointCloud(proto.Message):
    point = proto.Field(GMesPoints, number=1)
    # header = proto.Field(GMesHeader, number=2)
    idx = proto.Field(proto.INT32, number=2)
    @staticmethod
    def from_numpy(data):
        return GMesPointCloud(point=GMesPoints(x=data[0][:, 0].tolist(), 
                                               y=data[0][:, 1].tolist(), 
                                               z=data[0][:, 2].tolist()), 
                              idx=data[1])
    @staticmethod
    def to_numpy(data):
        return (np.array([data.point.x, data.point.y, data.point.z]).T, 
                data.idx)
class GMesTrimesh(proto.Message):
    vertex = proto.Field(GMesPoints, number=1)
    triangle = proto.Field(GMesFaces, number=2)
    idx = proto.Field(proto.INT32, number=3)
    @staticmethod
    def from_numpy(data):
        return GMesTrimesh(vertex=GMesPoints(x=data[0][:, 0].tolist(), 
                                             y=data[0][:, 1].tolist(), 
                                             z=data[0][:, 2].tolist()), 
                          triangle=GMesFaces(a=data[1][:, 0].tolist(), 
                                             b=data[1][:, 1].tolist(), 
                                             c=data[1][:, 2].tolist()), 
                          idx=data[2])
    @staticmethod
    def to_numpy(data):
        return (np.array([data.vertex.x, data.vertex.y, data.vertex.z]).T, 
                np.array([data.triangle.a, data.triangle.b, data.triangle.c]).T,
                data.idx)
    
class GMesImage(proto.Message):
    data = proto.Field(proto.BYTES, number=1)  
    height = proto.Field(proto.INT32, number=2)  # H
    width = proto.Field(proto.INT32, number=3)  # W
    channels = proto.Field(proto.INT32, number=4)  # C
    idx = proto.Field(proto.INT32, number=5)  

    @staticmethod
    def from_numpy(data):
        img, idx = data
        assert isinstance(img, np.ndarray)
        return GMesImage(
            data=img.tobytes(),
            height=img.shape[0],
            width=img.shape[1],
            channels=img.shape[2] if len(img.shape) > 2 else 1,
            idx=idx
        )

    @staticmethod
    def to_numpy(data):
        img_array = np.frombuffer(data.data, dtype=np.uint8)
        if data.channels > 1:
            img_array = img_array.reshape(data.height, data.width, data.channels)
        else:
            img_array = img_array.reshape(data.height, data.width)
        return img_array, data.idx
    
class GMesssage(proto.Message):
    pointcloud = proto.RepeatedField(GMesPointCloud, number=1)
    trimesh = proto.RepeatedField(GMesTrimesh, number=2)
    image = proto.RepeatedField(GMesImage, number=3)

if __name__ == '__main__':
    pass

