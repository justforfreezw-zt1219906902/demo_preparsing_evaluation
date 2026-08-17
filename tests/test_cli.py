from openscan_eval.cli import build_parser, _context


def test_full_resolution_keeps_crop_and_disables_only_resize(tmp_path):
    (tmp_path/"positions.csv").write_text("image,position_index,phi_deg,theta_deg\n")
    args=build_parser().parse_args(["preprocess","--dataset",str(tmp_path),"--full-resolution"])
    config,_,_=_context(args)
    assert config["crop"]["enabled"] is True
    assert config["crop"]["resize"]["enabled"] is False

