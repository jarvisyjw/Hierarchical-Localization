from ..utils.base_model import BaseModel
import cv2

EPS = 1e-6


class DoG(BaseModel):
    default_conf = {
        'options': {
            'first_octave': 0,
            'peak_threshold': 0.01,
        },
        'descriptor': 'rootsift',
        'max_keypoints': -1,
        'patch_size': 32,
        'mr_size': 12,
    }
    required_inputs = ['image']
    detection_noise = 1.0
    max_batch_size = 1024

    def _init(self, conf):
        self.orb =  cv2.ORB_create() 

    def _forward(self, data):
        image = data['image']
        image_np = image.cpu().numpy()[0, 0]
        assert image.shape[1] == 1
        assert image_np.min() >= -EPS and image_np.max() <= 1 + EPS
        
        keypoints, descriptors = self.orb.detectAndCompute(image_np)
        scales = keypoints[:, 2]
      #   oris = np.rad2deg(keypoints[:, 3])

        keypoints = torch.from_numpy(keypoints[:, :2])  # keep only x, y
        scales = torch.from_numpy(scales)
        oris = torch.from_numpy(oris)
        scores = torch.from_numpy(scores)

        if self.conf['max_keypoints'] != -1:
            # TODO: check that the scores from PyCOLMAP are 100% correct,
            # follow https://github.com/mihaidusmanu/pycolmap/issues/8
            indices = torch.topk(scores, self.conf['max_keypoints'])
            keypoints = keypoints[indices]
            scales = scales[indices]
            oris = oris[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]

        return {
            'keypoints': keypoints[None],
            'scales': scales[None],
            'oris': oris[None],
            'scores': scores[None],
            'descriptors': descriptors.T[None],
        }