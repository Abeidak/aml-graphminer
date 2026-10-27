"""Model and loss definitions — need torch + torch_geometric (skipped otherwise)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from aml_graphminer.detectors.gnn import GNNConfig  # noqa: E402
from aml_graphminer.detectors.gnn.losses import (  # noqa: E402
    binary_focal_loss_with_logits,
    build_criterion,
)
from aml_graphminer.detectors.gnn.models import (  # noqa: E402
    EdgeGNN,
    NodeGNN,
    count_parameters,
    make_conv,
    model_summary,
)
from aml_graphminer.exceptions import ConfigError  # noqa: E402

pytestmark = pytest.mark.gnn

N_NODE, N_EDGE_FEAT = 12, 7


@pytest.fixture()
def batch():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(N_NODE, 5, generator=gen)
    edge_index = torch.randint(0, N_NODE, (2, 40), generator=gen)
    label_index = torch.randint(0, N_NODE, (2, 9), generator=gen)
    edge_feat = torch.randn(9, N_EDGE_FEAT, generator=gen)
    return x, edge_index, label_index, edge_feat


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat"])
def test_edge_gnn_forward_shape_and_determinism_in_eval(kind, batch):
    x, edge_index, label_index, edge_feat = batch
    model = EdgeGNN(kind, 5, N_EDGE_FEAT, hidden=16, layers=2, dropout=0.3).eval()
    out = model(x, edge_index, label_index, edge_feat)
    assert out.shape == (9,) and torch.isfinite(out).all()
    assert torch.allclose(out, model(x, edge_index, label_index, edge_feat))


def test_dropout_is_active_in_train_mode_only(batch):
    x, edge_index, label_index, edge_feat = batch
    model = EdgeGNN("sage", 5, N_EDGE_FEAT, hidden=16, dropout=0.5)
    model.train()
    a, b = (
        model(x, edge_index, label_index, edge_feat),
        model(x, edge_index, label_index, edge_feat),
    )
    assert not torch.allclose(a, b)


def test_encode_decode_equals_forward(batch):
    x, edge_index, label_index, edge_feat = batch
    model = EdgeGNN("gcn", 5, N_EDGE_FEAT, hidden=8).eval()
    h = model.encode(x, edge_index)
    assert h.shape == (N_NODE, 8)
    assert torch.allclose(
        model.decode(h, label_index, edge_feat), model(x, edge_index, label_index, edge_feat)
    )


def test_from_config_and_summary():
    cfg = GNNConfig(kind="gat", hidden=16, gat_heads=2, num_neighbors=[3, 3])
    model = EdgeGNN.from_config(cfg, 5, N_EDGE_FEAT)
    assert model.kind == "gat" and count_parameters(model) > 0
    assert model_summary(model)["class"] == "EdgeGNN"
    assert isinstance(NodeGNN.from_config(cfg, 5), NodeGNN)


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat"])
def test_node_gnn_forward(kind, batch):
    x, edge_index, *_ = batch
    out = NodeGNN(kind, 5, hidden=8, layers=2).eval()(x, edge_index)
    assert out.shape == (N_NODE,)


def test_make_conv_validation():
    with pytest.raises(ConfigError, match="unknown GNN kind"):
        make_conv("transformer", 4, 4)
    with pytest.raises(ConfigError, match="divisible"):
        make_conv("gat", 4, 6, heads=4)


def test_focal_loss_reduces_to_scaled_bce_and_downweights_easy_examples():
    logits = torch.tensor([3.0, -3.0, 0.5, -0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert binary_focal_loss_with_logits(
        logits, targets, gamma=0.0, alpha=0.5
    ).item() == pytest.approx(0.5 * bce.item(), rel=1e-5)
    easy = binary_focal_loss_with_logits(
        torch.tensor([6.0]), torch.tensor([1.0]), gamma=2.0, alpha=0.5
    )
    hard = binary_focal_loss_with_logits(
        torch.tensor([-6.0]), torch.tensor([1.0]), gamma=2.0, alpha=0.5
    )
    assert easy.item() < hard.item()


def test_build_criterion_variants():
    logits, targets = torch.tensor([0.2, -0.4, 1.1]), torch.tensor([1.0, 0.0, 1.0])
    plain = build_criterion(GNNConfig(pos_weight=1.0))(logits, targets)
    weighted = build_criterion(GNNConfig(pos_weight=30.0))(logits, targets)
    assert weighted.item() > plain.item()
    assert build_criterion(GNNConfig(loss="focal"))(logits, targets).item() > 0
