import time
from config import *
from server import Server
from device import Device
from rsu import RSU
from shared_logger import logger

def main():
    server = Server(SERVER_PORT, expected_rsus=2)
    server.start()

    r1 = RSU("RSU_1", C1_RSU, CLUSTER1_PORTS, SERVER_PORT)
    r2 = RSU("RSU_2", C2_RSU, CLUSTER2_PORTS, SERVER_PORT)
    r1.start()
    r2.start()

    devices = [
        Device("C1_D1", 5001, C1_RSU, device_id=0),
        Device("C1_D2", 5002, C1_RSU, device_id=1),
        Device("C2_D1", 6001, C2_RSU, device_id=2),
        Device("C2_D2", 6002, C2_RSU, device_id=3)
    ]

    for d in devices:
        d.start()

    active = True
    while active:
        time.sleep(5)
        if len(logger.global_table.rows) >= TOTAL_ROUNDS:
            active = False

    print("\nTraining Complete! Saving logs to 'training_logs.txt'...")
    logger.save_logs("training_logs.txt")
    print("Logs successfully saved. Exiting.")

if __name__ == "__main__":
    main()