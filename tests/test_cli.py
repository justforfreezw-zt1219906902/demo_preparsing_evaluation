from openscan_eval.cli import build_parser, _context


def test_full_resolution_entry_point_option():
    args=build_parser().parse_args(["all","--full-resolution"])
    assert args.command=="all" and args.full_resolution is True


def test_full_resolution_disables_crop(tmp_path):
    (tmp_path/"positions.csv").write_text("image,position_index,phi_deg,theta_deg\n")
    args=build_parser().parse_args(["preprocess","--dataset",str(tmp_path),"--full-resolution"])
    config,_,_=_context(args)
    assert config["crop"]["enabled"] is False and config["crop"]["output_size"] is None
