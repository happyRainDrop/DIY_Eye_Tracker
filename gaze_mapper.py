import numpy as np
import cv2


class GazeMapper:

    def __init__(self, calibration_file, neutral_pupil):
        self.neutral = np.array(neutral_pupil, dtype=np.float32)
        data=np.load(calibration_file)
        if data.shape[0] < 6: raise RuntimeError("Need more calibration points")

        # calibration format:
        #
        # vector_x
        # vector_y
        # world_x
        # world_y

        vectors=data[:,0:2].astype(np.float32)
        world=data[:,2:4].astype(np.float32)

        # create polynomial features:
        # x,y
        # x²,y²,xy
        X=self.features(vectors)

        # solve:
        # world_x = model_x(features)
        # world_y = model_y(features)
        self.model_x,_,_,_=np.linalg.lstsq(X, world[:,0], rcond=None)
        self.model_y,_,_,_=np.linalg.lstsq(X, world[:,1], rcond=None)


    def project_to_bounds(self, x, y, width, height):

        cx = width / 2.0
        cy = height / 2.0

        dx = x - cx
        dy = y - cy

        if 0 <= x < width and 0 <= y < height:
            return int(x), int(y)

        scales = []
        if dx != 0:
            scales.append((0 - cx) / dx)
            scales.append((width - 1 - cx) / dx)
        if dy != 0:
            scales.append((0 - cy) / dy)
            scales.append((height - 1 - cy) / dy)

        candidates = []

        for t in scales:
            if t <= 0:
                continue
            px = cx + t * dx
            py = cy + t * dy
            if 0 <= px < width and 0 <= py < height:
                candidates.append((t, px, py))

        if candidates:
            _, px, py = min(candidates, key=lambda c: c[0])
            return int(px), int(py)

        return (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))

    def features(self,points):

        x=points[:,0]
        y=points[:,1]


        return np.column_stack(
            [
                np.ones(len(points)),
                x,
                y,
                x*x,
                y*y,
                x*y
            ]
        )

    def pupil_to_vector(self, pupil):
        return np.array([pupil[0]-self.neutral[0], pupil[1]-self.neutral[1]], dtype=np.float32)


    def map_pupil_to_world(self,pupil):

        if pupil is None: return None

        vector=self.pupil_to_vector(pupil)
        X=self.features(vector.reshape(1,2))
        wx=float(X @ self.model_x)
        wy=float(X @ self.model_y)

        image_size = (640, 480)  # World camera wxh
        if image_size is not None:
            width, height = image_size
            return self.project_to_bounds(wx,wy,width,height)

        return (int(wx),int(wy))