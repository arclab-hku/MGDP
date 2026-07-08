from g_basic import Client

if __name__ == "__main__":
    client = Client(address="tcp://localhost:5555")
    try:
        while True:
            message = input("Enter message to send (type 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            response = client.send(bytes(message, encoding='utf-8'))
            if response:
                print(f"Server response: {response}")
            else:
                print("No response received.")
    except KeyboardInterrupt:
        pass
    finally:
        client.close()