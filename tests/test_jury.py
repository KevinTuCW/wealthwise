import wealthwise.crosscheck.jury as jury
from wealthwise.config import Settings
from wealthwise.llm import OpenAICompatibleModelClient


def test_build_jury_clients_from_settings(monkeypatch):
    captured = []

    def fake_init(self, name, model, base_url, api_key):
        self.name = name
        self._model = model
        captured.append((name, model, base_url, api_key))

    monkeypatch.setattr(OpenAICompatibleModelClient, "__init__", fake_init)

    s = Settings(glm_api_key="k1", llm_model="glm-4.7",
                 glm_base_url="https://api.z.ai/api/paas/v4/",
                 siliconflow_api_key="k2",
                 crosscheck_model="deepseek-ai/DeepSeek-V3",
                 siliconflow_base_url="https://api.siliconflow.com/v1")
    clients = jury.build_jury_clients(s)

    assert [c.name for c in clients] == ["glm-4.7", "deepseek-ai/DeepSeek-V3"]
    assert captured == [
        ("glm-4.7", "glm-4.7", "https://api.z.ai/api/paas/v4/", "k1"),
        ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3",
         "https://api.siliconflow.com/v1", "k2"),
    ]
