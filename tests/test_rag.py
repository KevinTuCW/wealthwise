import pytest

from wealthwise.rag.embed import LocalHashingEmbedder, Embedder
from wealthwise.rag.store import Doc, InMemoryVectorStore, Retriever
from wealthwise.rag.corpus import load_policy_retriever, load_research_retriever
from wealthwise.config import Settings
from wealthwise.rag.backends import build_embedder


# --------------------------------------------------------------------------- #
# Embedder                                                                     #
# --------------------------------------------------------------------------- #

def test_embedder_satisfies_protocol():
    e = LocalHashingEmbedder(dim=64)
    assert isinstance(e, Embedder)


def test_embedder_vector_length_matches_dim():
    e = LocalHashingEmbedder(dim=64)
    v = e.embed("股票 风险 收益")
    assert len(v) == 64


def test_embedder_is_l2_normalized():
    e = LocalHashingEmbedder(dim=256)
    v = e.embed("适当性匹配 投资者保护")
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_embedder_is_deterministic():
    e = LocalHashingEmbedder(dim=128)
    text = "黄金 ETF 避险资产"
    assert e.embed(text) == e.embed(text)


# --------------------------------------------------------------------------- #
# InMemoryVectorStore                                                          #
# --------------------------------------------------------------------------- #

def test_store_satisfies_retriever_protocol():
    store = InMemoryVectorStore(LocalHashingEmbedder(dim=256))
    assert isinstance(store, Retriever)


def test_store_retrieves_most_similar_first():
    store = InMemoryVectorStore(LocalHashingEmbedder(dim=256))
    store.add([
        Doc(id="risk-disclosure", text="风险揭示要求 投资者签字 本金损失", meta={}),
        Doc(id="gold-etf", text="黄金ETF避险配置 低相关性 组合分散", meta={}),
    ])
    top = store.search("风险揭示书签署要求", k=1)
    assert top[0].id == "risk-disclosure"


def test_store_retrieves_finance_doc():
    store = InMemoryVectorStore(LocalHashingEmbedder(dim=256))
    store.add([
        Doc(id="leverage", text="杠杆融资融券 强制平仓 追加保证金 风险", meta={}),
        Doc(id="reits", text="基础设施REITs 高速公路 产业园区 分红", meta={}),
    ])
    top = store.search("融资买入 强制平仓机制", k=1)
    assert top[0].id == "leverage"


def test_search_k_limits_results():
    store = InMemoryVectorStore(LocalHashingEmbedder(dim=128))
    store.add([Doc(id=str(i), text=f"投资标的 {i} 研报分析", meta={}) for i in range(5)])
    assert len(store.search("研报", k=3)) == 3


# --------------------------------------------------------------------------- #
# Corpus loaders                                                               #
# --------------------------------------------------------------------------- #

def test_load_policy_retriever_loads_and_retrieves():
    r = load_policy_retriever("data/samples", LocalHashingEmbedder(dim=256))
    top = r.search("投资者 风险承受能力 适当性 匹配", k=2)
    ids = {d.id for d in top}
    # should surface the suitability or risk-disclosure clause
    assert ids & {"suitability-match", "risk-disclosure", "investor-classification"}


def test_load_policy_retriever_no_guaranteed_return():
    r = load_policy_retriever("data/samples", LocalHashingEmbedder(dim=256))
    top = r.search("不得 承诺 收益 固定回报 保本", k=1)
    assert top[0].id == "no-guaranteed-return"


def test_load_policy_retriever_fx_query():
    r = load_policy_retriever("data/samples", LocalHashingEmbedder(dim=256))
    top = r.search("跨境投资 汇率风险 通道 资金", k=1)
    assert top[0].id == "fx-cross-border-risk"


def test_load_research_retriever_loads_and_retrieves():
    r = load_research_retriever("data/samples", LocalHashingEmbedder(dim=256))
    top = r.search("黄金 配置 避险资产 ETF", k=2)
    ids = {d.id for d in top}
    assert ids & {"gold-commodity-2025"}


def test_load_research_retriever_equity_query():
    r = load_research_retriever("data/samples", LocalHashingEmbedder(dim=256))
    top = r.search("招商银行 ROE 零售贷款", k=1)
    assert top[0].id == "600036-cmb-2024"


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #

def test_local_provider_builds_hashing_embedder():
    s = Settings(embed_provider="local", embed_dim=128)
    e = build_embedder(s)
    assert isinstance(e, LocalHashingEmbedder)
    assert e.dim == 128


def test_siliconflow_provider_builds_real_embedder_without_network(monkeypatch):
    from wealthwise.rag import backends
    captured = {}

    class _Fake:
        def __init__(self, model, base_url, api_key, dim):
            captured.update(model=model, base_url=base_url, api_key=api_key, dim=dim)
            self.dim = dim

    monkeypatch.setattr(backends, "SiliconFlowEmbedder", _Fake)
    s = Settings(embed_provider="siliconflow", embed_model="Qwen/Qwen3-Embedding-8B",
                 embed_dim=1024, siliconflow_api_key="k",
                 siliconflow_base_url="https://api.siliconflow.com/v1")
    e = build_embedder(s)
    assert e.dim == 1024
    assert captured == {"model": "Qwen/Qwen3-Embedding-8B",
                        "base_url": "https://api.siliconflow.com/v1", "api_key": "k", "dim": 1024}
