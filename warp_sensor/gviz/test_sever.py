from g_basic import Server

if __name__ == "__main__":
    server = Server(address="tcp://*:5555")
    server.start()
    print("Server started. Press Ctrl+C to stop.")
    while not server.running:
        time.sleep(0.1)
    try:
        while True:
            least_msg = server.get_message()
            if least_msg != None:
                print(f"Received message: {least_msg}")
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop()