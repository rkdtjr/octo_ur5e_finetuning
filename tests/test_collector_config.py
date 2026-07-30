from pathlib import Path
import yaml,pytest
from octo_ur5e_collector.core.config import load_config,ConfigError
BASE=Path(__file__).parents[1]/"collector/config/collector.yaml"
def test_valid(): assert load_config(BASE).data["schema_version"]==2
@pytest.mark.parametrize("change",["unknown","polarity","mask","joints","rate","timeout"])
def test_invalid(tmp_path,change):
    d=yaml.safe_load(BASE.read_text())
    if change=="unknown":d["bogus"]=1
    if change=="polarity":d["gripper"]["output_value_for_closed"]=1.0
    if change=="mask":d["action_contract"]["normalization_mask"]=[True]
    if change=="joints":d["robot"]["joint_names"]=["x"]*6
    if change=="rate":d["sampling"]["demonstration_rate_hz"]=0
    if change=="timeout":d["gripper"]["command_timeout_sec"]=-1
    p=tmp_path/"c.yaml";p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError):load_config(p)
