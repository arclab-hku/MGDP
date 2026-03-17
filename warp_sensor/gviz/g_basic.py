import zmq
import threading
from queue import Queue, Empty
from typing import Optional, Any, List
import logging
from collections import deque
from threading import Lock
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Server:
    """
    Enhanced ZeroMQ server with message queue and async processing for binary messages
    """
    def __init__(self, address: str = "tcp://*:5555", max_recent_messages: int = 1000):
        """
        Initialize server with specified address

        Args:
            address: ZMQ binding address
            max_recent_messages: Maximum number of recent messages to keep in memory
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(address)
        
        # High-water mark for better flow control
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        self.socket.setsockopt(zmq.SNDHWM, 1000)
        
        self.running = False
        self.message_queue = Queue()
        self.recent_messages = deque(maxlen=max_recent_messages)
        self.message_lock = Lock()
        self.worker_threads = []
        self.num_workers = 16
        
        # Message statistics
        self.messages_received = 0
        self.messages_processed = 0
        self.stats_lock = Lock()

    def start(self):
        """Start server and worker threads"""
        self.running = True
        
        # Start message receiving thread
        self.receiver_thread = threading.Thread(target=self._receive_messages)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

        # Start worker threads
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._process_messages)
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)

        logger.info("Server started successfully")

    # TODO: Add threadpool for processing messages
    def _receive_messages(self):
        """Main message receiving loop"""
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        
        while self.running:
            try:
                # Use poller with timeout to reduce CPU usage
                if poller.poll(100):  # 100ms timeout
                    message = self.socket.recv()
                    self.message_queue.put(message)
                    with self.message_lock:
                        self.recent_messages.append(message)
                    with self.stats_lock:
                        self.messages_received += 1
                else:
                    # Sleep briefly if no messages
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                time.sleep(0.1)  # Prevent tight loop on error

    def _process_messages(self):
        """Worker thread for processing messages"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1.0)
                self.socket.send(message)
                self.message_queue.task_done()
                with self.stats_lock:
                    self.messages_processed += 1
            except Empty:
                time.sleep(0.01)  # Reduce CPU usage when queue is empty
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                try:
                    self.socket.send(b"Error processing request")
                except:
                    pass
                time.sleep(0.1)  # Prevent tight loop on error

    def get_message(self, pop: bool = True) -> Optional[bytes]:
        """
        Get the most recent message received by the server
        
        Args:
            pop: If True, remove the message after retrieving it
        
        Returns:
            Optional[bytes]: The most recent message, or None if no messages are available
        """
        with self.message_lock:
            try:
                if not self.recent_messages:
                    return None
                if pop:
                    return self.recent_messages.pop()
                return self.recent_messages[-1]
            except IndexError:
                return None

    def get_messages(self, count: int = 1, pop: bool = True) -> List[bytes]:
        """
        Get the most recent n messages received by the server
        
        Args:
            count: Number of recent messages to retrieve
            pop: If True, remove the messages after retrieving them
            
        Returns:
            List[bytes]: List of recent messages, newest first
        """
        with self.message_lock:
            if not self.recent_messages:
                return []
            
            if pop:
                messages = []
                for _ in range(min(count, len(self.recent_messages))):
                    try:
                        messages.append(self.recent_messages.pop())
                    except IndexError:
                        break
                return messages
            else:
                return list(self.recent_messages)[-count:]

    def get_queue_status(self) -> dict:
        """
        Get current status of the message queue and processing statistics
        
        Returns:
            dict: Dictionary containing queue and processing statistics
        """
        with self.stats_lock:
            status = {
                'queue_size': self.message_queue.qsize(),
                'messages_received': self.messages_received,
                'messages_processed': self.messages_processed,
                'messages_pending': self.messages_received - self.messages_processed,
                'recent_messages_count': len(self.recent_messages),
                'is_running': self.running,
                'active_workers': sum(1 for t in self.worker_threads if t.is_alive()),
                'total_workers': self.num_workers
            }
        return status

    def reset_statistics(self):
        """Reset all message statistics counters"""
        with self.stats_lock:
            self.messages_received = 0
            self.messages_processed = 0

    def stop(self):
        """Stop server and cleanup resources"""
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'receiver_thread'):
            self.receiver_thread.join()
        for worker in self.worker_threads:
            worker.join()

        # Cleanup ZMQ resources
        self.socket.close()
        self.context.term()
        logger.info("Server stopped")

class Client:
    """
    Enhanced ZeroMQ client with automatic reconnection and timeout handling for binary messages
    """
    def __init__(self, address: str = "tcp://localhost:5555", timeout: int = 5000):
        """
        Initialize client with specified address and timeout

        Args:
            address: Server address to connect to
            timeout: Timeout in milliseconds
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.address = address
        self.socket.connect(address)

    def send(self, message: bytes) -> Optional[bytes]:
        """
        Send binary message to server and receive response

        Args:
            message: Binary message content

        Returns:
            Optional[bytes]: Server response or None if error occurs
        """
        try:
            self.socket.send(message)
            return self.socket.recv()
        except zmq.ZMQError as e:
            logger.error(f"ZMQ Error: {e}")
            self._reconnect()
            return None
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None

    def _reconnect(self):
        """Attempt to reconnect to server"""
        try:
            self.socket.close()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.address)
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    def close(self):
        """Close client connection and cleanup"""
        self.socket.close()
        self.context.term()