import http.client
import json
import unittest

from stratarag import Agent, Knowledge, Memory
from stratarag.dashboard import serve


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kb = Knowledge.from_texts(
            ["Refunds are accepted within 30 days of purchase."])
        cls.agent = Agent(model="echo", knowledge=kb,
                          memory=Memory(semantic=True))
        cls.server = serve(cls.agent, host="127.0.0.1", port=0, block=False)
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _req(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path,
                     body=json.dumps(body) if body else None,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read().decode()
        conn.close()
        return resp.status, data

    def test_serves_page_and_health(self):
        status, html = self._req("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("stratarag playground", html)
        status, health = self._req("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(health)["has_knowledge"])

    def test_chat_endpoint_returns_result_shape(self):
        status, data = self._req("POST", "/api/chat",
                                 {"message": "refund window?", "user_id": "u1"})
        self.assertEqual(status, 200)
        payload = json.loads(data)
        for key in ("answer", "confidence", "gated", "sources", "memory", "trace"):
            self.assertIn(key, payload)
        self.assertIn("Refunds", payload["answer"])

    def test_chat_validates_input(self):
        status, data = self._req("POST", "/api/chat", {"message": "  "})
        self.assertEqual(status, 400)
        status, _ = self._req("POST", "/api/unknown", {})
        self.assertEqual(status, 404)

    def test_remember_endpoint(self):
        status, _ = self._req("POST", "/api/remember",
                              {"fact": "User prefers brevity", "user_id": "u1"})
        self.assertEqual(status, 200)
        facts = self.agent.memory.semantic.all("u1")
        self.assertTrue(any("brevity" in f.content for f in facts))


if __name__ == "__main__":
    unittest.main()
