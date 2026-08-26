import unittest

import torch

from crypto_protocol import Authority
from device import Device
from rsu import RSU
from server import Server


class GlobalUpdateSecurityTests(unittest.TestCase):
    def setUp(self):
        self.authority = Authority()
        for name in ["Server", "RSU_0_Central", "C0_V1"]:
            self.authority.enroll_mvd(name)
        self.server_id = self.authority.register("Server")
        self.rsu_id = self.authority.register("RSU_0_Central")
        self.vehicle_id = self.authority.register("C0_V1")
        self.weights = {"weight": torch.tensor([1.0, 2.0])}

    def test_server_to_rsu_and_rsu_to_vehicle_are_authenticated(self):
        server = Server.__new__(Server)
        server.security_enabled = True
        server.security_authority = self.authority
        server.signer = self.server_id
        server_msg = server._build_global_message(
            "RSU_0_Central", 2, self.weights)

        rsu = RSU.__new__(RSU)
        rsu.name = "RSU_0_Central"
        rsu.security_enabled = True
        rsu.security_authority = self.authority
        rsu.signer = self.rsu_id
        decoded = rsu._decode_server_global(server_msg)
        self.assertTrue(torch.equal(
            decoded["weight"], self.weights["weight"]))

        vehicle_msg = rsu._build_vehicle_global("C0_V1", 2, decoded)
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.rsu_name = "RSU_0_Central"
        device.security_enabled = True
        device.security_authority = self.authority
        device.signer = self.vehicle_id
        final = device._decode_rsu_global(vehicle_msg)
        self.assertTrue(torch.equal(
            final["weight"], self.weights["weight"]))

    def test_unsigned_or_tampered_global_update_is_rejected(self):
        device = Device.__new__(Device)
        device.name = "C0_V1"
        device.rsu_name = "RSU_0_Central"
        device.security_enabled = True
        device.security_authority = self.authority
        device.signer = self.vehicle_id
        unsigned = {
            "type": "GLOBAL_UPDATE",
            "sender": "RSU_0_Central",
            "round": 2,
        }
        self.assertIsNone(device._decode_rsu_global(unsigned))

        rsu = RSU.__new__(RSU)
        rsu.name = "RSU_0_Central"
        rsu.security_enabled = True
        rsu.security_authority = self.authority
        rsu.signer = self.rsu_id
        tampered = rsu._build_vehicle_global("C0_V1", 2, self.weights)
        ciphertext = bytearray(tampered["ciphertext"])
        ciphertext[0] ^= 1
        tampered["ciphertext"] = bytes(ciphertext)
        self.assertIsNone(device._decode_rsu_global(tampered))


if __name__ == "__main__":
    unittest.main()
