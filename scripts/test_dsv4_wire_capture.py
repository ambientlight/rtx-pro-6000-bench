import ast
import pathlib
import socket
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import dsv4_wire_capture as capture


def packet(source="172.18.0.2", destination="172.18.0.1", sport=8010, dport=43210,
           vlan=False, protocol=6):
    ethernet = bytes(12) + (b"\x81\x00\x00\x01\x08\x00" if vlan else b"\x08\x00")
    ipv4 = bytearray(20)
    ipv4[0] = 0x45
    ipv4[9] = protocol
    ipv4[12:16] = socket.inet_aton(source)
    ipv4[16:20] = socket.inet_aton(destination)
    return ethernet + ipv4 + struct.pack("!HH", sport, dport) + bytes(16)


class PacketFilterTest(unittest.TestCase):
    def setUp(self):
        self.ips = {socket.inet_aton("172.18.0.2"), socket.inet_aton("127.0.0.1")}

    def test_request_and_response(self):
        self.assertTrue(capture.packet_matches_http(packet(), self.ips, 8010))
        self.assertTrue(capture.packet_matches_http(packet(source="172.18.0.1",
            destination="172.18.0.2", sport=43210, dport=8010), self.ips, 8010))

    def test_loopback_smoke_is_captured_only_when_enabled(self):
        frame = packet(source="127.0.0.1", destination="127.0.0.1")
        self.assertTrue(capture.packet_matches_http(frame, self.ips, 8010))
        self.assertFalse(capture.packet_matches_http(frame, {socket.inet_aton("172.18.0.2")}, 8010))

    def test_vlan_and_truncated_frames(self):
        frame = packet(vlan=True)
        self.assertTrue(capture.packet_matches_http(frame, self.ips, 8010))
        for length in range(42):
            self.assertFalse(capture.packet_matches_http(frame[:length], self.ips, 8010))

    def test_unrelated_traffic_is_excluded(self):
        self.assertFalse(capture.packet_matches_http(packet(sport=8001), self.ips, 8010))
        self.assertFalse(capture.packet_matches_http(packet(source="172.18.0.9"), self.ips, 8010))
        self.assertFalse(capture.packet_matches_http(packet(protocol=17), self.ips, 8010))

    def test_embedded_filter_is_the_tested_function(self):
        tree = ast.parse(capture.CONTAINER_CAPTURE_PROGRAM)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "packet_matches_http")
        scope = {"struct": struct}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "capture-filter", "exec"), scope)
        for frame in (packet(), packet(vlan=True), packet(source="127.0.0.1"), b"short"):
            self.assertEqual(scope["packet_matches_http"](frame, self.ips, 8010),
                             capture.packet_matches_http(frame, self.ips, 8010))


class KernelFilterTest(unittest.TestCase):
    def test_actual_kernel_filter_accepts_http_and_rejects_other_traffic(self):
        # Linux socket filters also work on local datagrams: exercise the real
        # bytecode without root, GPU resources, or a packet-filter emulator.
        sender, receiver = socket.socketpair(type=socket.SOCK_DGRAM)
        with sender, receiver:
            receiver.settimeout(0.01)
            capture.attach_kernel_http_filter(receiver, 8010)
            frames = [
                (packet(), True), (packet(vlan=True), True),
                (packet(source="127.0.0.1", destination="127.0.0.1"), True),
                (packet(sport=43210, dport=8010), True),
                (packet(sport=43210, dport=8010, vlan=True), True),
                (packet(sport=8001), False), (packet(protocol=17), False),
                (packet(sport=8001, vlan=True), False),
                (packet(protocol=17, vlan=True), False), (b"short", False),
            ]
            # IPv4 options move the TCP header; BPF must use IHL, not a fixed
            # offset. Existing user-space filtering already handles this too.
            options = bytearray(packet())
            options[14] = 0x46
            options[34:34] = bytes(4)
            frames.append((bytes(options), True))
            for frame, accepted in frames:
                with self.subTest(frame=frame, accepted=accepted):
                    sender.send(frame)
                    if accepted:
                        self.assertEqual(receiver.recv(65536), frame)
                    else:
                        with self.assertRaises(socket.timeout):
                            receiver.recv(65536)

    def test_embedded_filter_is_same_tested_bytecode(self):
        tree = ast.parse(capture.CONTAINER_CAPTURE_PROGRAM)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "kernel_http_filter")
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "kernel-filter", "exec"), scope)
        self.assertEqual(scope["kernel_http_filter"](8010), capture.kernel_http_filter(8010))
        for invalid in (0, -1, 65536):
            with self.assertRaises(ValueError):
                capture.kernel_http_filter(invalid)


class AuthenticatedLoggingTest(unittest.TestCase):
    def test_deployment_identity_does_not_request_secret_configuration(self):
        result = Mock(stdout='{"id":"container-id","image_id":"sha256:abc","image_tag":"fixture:tag","init_pid":123,"started_at":"2026-09-15T00:00:00Z"}')
        with patch.object(capture.subprocess, "run", return_value=result) as run:
            identity = capture.container_identity("fixture")
        self.assertEqual(identity["init_pid"], 123)
        command = run.call_args.args[0]
        self.assertNotIn(".Config.Env", command[3])
        self.assertNotIn(".Mounts", command[3])
        self.assertEqual(command[-1], "fixture")

    def test_private_key_is_sent_but_not_returned(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = pathlib.Path(temporary) / "api-key"
            key_file.write_text("fixture-secret\n")
            response = Mock(status=200)
            response.read.return_value = b'{"status":"ok"}'
            opener = Mock()
            opener.return_value.__enter__ = Mock(return_value=response)
            opener.return_value.__exit__ = Mock(return_value=False)
            with patch.object(capture.urllib.request, "urlopen", opener):
                result = capture.enable_native_logging("http://127.0.0.1:8010", key_file)
            request = opener.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer fixture-secret")
            self.assertTrue(result[0])
            self.assertNotIn("fixture-secret", result[1])

    def test_missing_and_invalid_keys_do_not_make_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = pathlib.Path(temporary) / "api-key"
            with patch.object(capture.urllib.request, "urlopen") as opener:
                self.assertFalse(capture.enable_native_logging("http://127.0.0.1:8010", key_file)[0])
                key_file.write_text("fixture-secret\ninjected")
                result = capture.enable_native_logging("http://127.0.0.1:8010", key_file)
                self.assertFalse(result[0])
                self.assertNotIn("fixture-secret", result[1])
                opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
