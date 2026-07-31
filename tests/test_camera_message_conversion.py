import numpy as np,pytest
from octo_ur5e_collector.ros_adapters.camera_video_recorder import image_message_to_bgr
class Msg:
    def __init__(self,encoding,data):self.encoding=encoding;self.data=data
class Bridge:
    def imgmsg_to_cv2(self,msg,desired_encoding):
        if desired_encoding=="bgr8" and msg.encoding=="rgb8":return msg.data[:,:,::-1]
        return msg.data
def test_bayer_and_rgb_are_both_accepted():
    raw=np.zeros((4,4),np.uint8)
    assert image_message_to_bgr(Bridge(),Msg("bayer_grbg8",raw)).shape==(4,4,3)
    rgb=np.zeros((4,4,3),np.uint8)
    assert image_message_to_bgr(Bridge(),Msg("rgb8",rgb)).shape==(4,4,3)
def test_three_channel_bayer_is_rejected_clearly():
    with pytest.raises(ValueError,match="one channel"):
        image_message_to_bgr(Bridge(),Msg("bayer_grbg8",np.zeros((4,4,3),np.uint8)))
