from openscan_eval.config import DEFAULT_CONFIG, load_config


def test_single_authoritative_default_config():
    assert str(DEFAULT_CONFIG).endswith("configs/default.yaml")


def test_user_crop_config_is_loaded(tmp_path):
    custom=tmp_path/"custom.yaml";custom.write_text("crop:\n  mode: manual\n  roi_xyxy: [0.1, 0.2, 0.7, 0.8]\n")
    config=load_config(custom)
    assert config["crop"]["roi_xyxy"]==[0.1,0.2,0.7,0.8]
    assert config["crop"]["resize"]["output_size"]==[1200,1000]
