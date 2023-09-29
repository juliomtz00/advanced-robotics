import threading
import random
import time

# Create a lock to ensure proper synchronization between threads
lock = threading.Lock()

# Shared variable to store the random number
shared_random_number = None

# Function to generate and display a random integer
def generate_and_display():
    global shared_random_number
    random_number = random.randint(1, 100)
    
    with lock:
        shared_random_number = random_number
    
    print(f"The sent number is: {random_number}")

# Function to receive and display the number
def receive_and_display():
    global shared_random_number
    with lock:
        random_number = shared_random_number
    
    print(f"The received number is: {random_number}")

# Create a thread for generating and displaying the random number
gen_thread = threading.Thread(target=generate_and_display)

# Create a thread for receiving and displaying the number
recv_thread = threading.Thread(target=receive_and_display)

# Start both threads
gen_thread.start()
recv_thread.start()

# Wait for both threads to finish
gen_thread.join()
recv_thread.join()
