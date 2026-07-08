from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from dataclasses import dataclass
from typing import Any, Optional
import threading
import time
import logging
from gviz.g_basic import *
from gviz.g_message import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO change name
@dataclass
class Message:
    data: Any
    msg_type: str

class ClientPool:
    def __init__(self, address: str, pool_size: int = 4, timeout: int = 5000):
        self.clients = deque(
            Client(address, timeout) for _ in range(pool_size)
        )
        self.lock = threading.Lock()

    def get_client(self) -> Client:
        with self.lock:
            client = self.clients.popleft()
            self.clients.append(client)
            return client

    def close_all(self):
        for client in self.clients:
            client.close()

class MessagePublisher:
    def __init__(self, address: str = "tcp://localhost:5555", 
                 max_workers: int = 4, 
                 client_pool_size: int = 4):
        self.client_pool = ClientPool(address, client_pool_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.msg_queue = Queue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.start()

    def _pack_message(self, msg: Message) -> bytes:
        """Pack message according to its type"""
        if msg.msg_type == "pointcloud":
            # TODO: Add support for multiple environments
            pcl = GMesPointCloud.from_numpy((msg.data, 0))
            message = GMesssage(pointcloud=[pcl])
            return GMesssage.serialize(message)
        elif msg.msg_type == "trimesh":
            mesh = GMesTrimesh.from_numpy((msg.data[0], msg.data[1], 0))
            message = GMesssage(trimesh=[mesh])
            return GMesssage.serialize(message)
        elif msg.msg_type == "image":
            # print(msg.data)
            img = GMesImage.from_numpy((msg.data[0], msg.data[1]))
            message = GMesssage(image=[img])
            return GMesssage.serialize(message)
        raise ValueError(f"Unknown message type: {msg.msg_type}")

    def _process_queue(self):
        """Worker thread to process messages in queue"""
        while self.running:
            try:
                if not self.msg_queue.empty():
                    msg = self.msg_queue.get()
                    self.executor.submit(self._send_message, msg)
                else:
                    time.sleep(0.01) 
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def _send_message(self, msg: Message):
        """Pack and send message"""
        try:
            packed_msg = self._pack_message(msg)
            client = self.client_pool.get_client()
            client.send(packed_msg)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def publish(self, data: Any, msg_type: str):
        """Add message to queue"""
        msg = Message(data=data, msg_type=msg_type)
        self.msg_queue.put(msg)

    def close(self):
        """Cleanup resources"""
        self.running = False
        self.worker_thread.join()
        self.executor.shutdown()
        self.client_pool.close_all()


if __name__ == "__main__":
    for i in range(10000):
        publisher = MessagePublisher()
        publisher.publish(np.random.rand(10000, 3), "pointcloud")
        # publisher.publish((np.random.rand(100, 3), np.random.randint(0, 100, (100, 3))), "trimesh")
        test_img = np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8)
        publisher.publish((test_img, 0), "image")
        time.sleep(0.1)
        publisher.close()