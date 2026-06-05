Yes. The basic pipeline is:

**camera 1 → pose keypoints → normalize → compare**
**camera 2 → pose keypoints → normalize → compare**

The important bit is **normalization**, otherwise different body sizes, camera angles, distances, and positions will ruin the comparison.

A good approach:

1. Detect pose for each person
   Use MediaPipe Pose or YOLO-pose.

2. Convert keypoints to a body-relative coordinate system
   For each frame:

   * set the hip centre as `(0, 0)`
   * scale everything by torso length or shoulder width
   * optionally rotate so shoulders/hips are aligned

3. Compare joint angles instead of raw points
   This is usually more robust. For example:

   * left elbow angle
   * right elbow angle
   * left knee angle
   * right knee angle
   * shoulder angle
   * hip angle

4. Compute a similarity score
   For each frame:

```text
similarity = 1 - average_normalized_angle_difference
```

Or with vectors:

```python
score = cosine_similarity(pose_vector_1, pose_vector_2)
```

5. Smooth over time
   Don’t compare just one frame. Compare over the last 0.5–2 seconds using a rolling average.

For “are they doing the same thing right now?”, compare frame-by-frame.

For “are they doing the same movement but one is delayed?”, use **Dynamic Time Warping**. That lets you compare motion sequences even if one person is slightly ahead/behind.

The pose vector might look like:

```python
pose_vector = [
    left_elbow_angle,
    right_elbow_angle,
    left_shoulder_angle,
    right_shoulder_angle,
    left_hip_angle,
    right_hip_angle,
    left_knee_angle,
    right_knee_angle,
]
```

Then:

```python
import numpy as np

def similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    diff = np.abs(a - b)
    diff = np.minimum(diff, 360 - diff)  # handle wraparound

    avg_diff = np.mean(diff)
    return max(0, 1 - avg_diff / 90)
```

So:

* `1.0` = very similar
* `0.7` = roughly similar
* `<0.5` = probably different

For your use case, I’d probably use:

**MediaPipe Pose → joint angles → rolling similarity score → optional DTW for movement sequences**

That will be much easier and more reliable than comparing the raw skeleton positions directly.
