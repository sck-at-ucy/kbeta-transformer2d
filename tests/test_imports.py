import mlx.core as mx

from kbeta_transformer2d import HeatDiffusionModel


def test_forward():
    ny = nx = 8
    model = HeatDiffusionModel(
        ny,
        nx,
        seq_len=4,
        num_heads=2,
        num_encoder_layers=1,
        mlp_dim=256,
        embed_dim=512,
        start_predicting_from=0,
        mask_type="block",
    )
    dummy = mx.zeros((1, 4, ny, nx))
    alphas = mx.ones((1,))
    out = model(dummy, alphas)
    assert out.shape == dummy.shape
