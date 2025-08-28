import tensorflow as tf
from tensorflow.keras import layers

class BasePoseCNN(tf.keras.Model):
    """
    1D CNN + Engineered Feature Layer (with Rule-Based Instructions & Masks)
    
    Preprocessing baked-in:
    - Mid-hip centering
    - Shoulder-width scaling
    - Joint masking (example mask for squats)

    Multi-head outputs:
    - Form score regression (learned, 0-1)
    - Instruction/Error type (rule-based)
    - Joint correction/confidence mask (rule-based)
    - Final 17-keypoint array with confidence as joint mask
    """

    def __init__(self, num_joints=17, num_instruction_classes=10, **kwargs):
        super(BasePoseCNN, self).__init__(**kwargs)
        self.num_joints = num_joints
        self.num_instruction_classes = num_instruction_classes

        # 🔹 Input reshape: (batch, 51) → (batch, 17, 3)
        self.reshape_layer = layers.Reshape((num_joints, 3))

        # 🔹 Example joint mask (for squats: ignore wrists & elbows)
        mask = [1, 1, 1,  # nose, eyes
                1, 1,    # ears
                1, 1,    # shoulders
                0, 0,    # elbows
                0, 0,    # wrists
                1, 1,    # hips
                1, 1,    # knees
                1, 1]    # ankles
        self.joint_mask = tf.constant(mask, dtype=tf.float32)

        # === CNN Backbone ===
        self.conv1 = layers.Conv1D(64, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv1D(64, 5, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv3 = layers.Conv1D(128, 3, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        self.global_pool = layers.GlobalAveragePooling1D()
        self.cnn_dense = layers.Dense(32, activation='relu', name="cnn_embedding")

        # === Engineered Features ===
        self.biomech_dense = layers.Dense(32, activation='relu', name="biomech_embedding")

        # === Fusion ===
        self.fusion_dense1 = layers.Dense(64, activation='relu')
        self.fusion_dropout1 = layers.Dropout(0.3)
        self.fusion_dense2 = layers.Dense(32, activation='relu')
        self.fusion_dropout2 = layers.Dropout(0.2)

        # === Multi-head outputs ===
        self.form_score_output = layers.Dense(1, activation='sigmoid', name='form_score')
        # Instruction & joint mask will be rule-based, included in final 17-keypoint output
        self.instruction_output = None
        self.joint_mask_output = None

    def preprocess_keypoints(self, keypoints):
        """
        Normalize keypoints:
        - Mid-hip centering
        - Shoulder-width scaling
        - Apply example joint mask
        Input: (batch, 17, 3) [y, x, conf]
        Output: (batch, 17, 3)
        """
        # --- Mid-hip centering ---
        left_hip = keypoints[:, 11, :2]
        right_hip = keypoints[:, 12, :2]
        mid_hip = (left_hip + right_hip) / 2.0
        keypoints_xy = keypoints[:, :, :2] - tf.expand_dims(mid_hip, axis=1)

        # --- Shoulder-width scaling ---
        left_shoulder = keypoints[:, 5, :2]
        right_shoulder = keypoints[:, 6, :2]
        shoulder_dist = tf.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
        shoulder_dist = tf.maximum(shoulder_dist, 1e-6)
        keypoints_xy /= tf.expand_dims(shoulder_dist, axis=1)

        # --- Reattach confidence ---
        keypoints_proc = tf.concat([keypoints_xy, tf.expand_dims(keypoints[:, :, 2], axis=-1)], axis=-1)

        # --- Apply joint mask ---
        mask = tf.reshape(self.joint_mask, (1, self.num_joints, 1))
        keypoints_proc = keypoints_proc * mask

        return keypoints_proc

    def compute_engineered_features(self, keypoints):
        """
        Compute biomechanical features (angles, alignments, symmetry, etc.)
        Input: (batch, 17, 3) normalized & masked
        Output: (batch, F) dense embedding
        """
        xy = keypoints[:, :, :2]

        # Example features
        def joint_angle(a, b, c):
            ba = a - b
            bc = c - b
            ba = tf.math.l2_normalize(ba, axis=-1)
            bc = tf.math.l2_normalize(bc, axis=-1)
            cos_angle = tf.reduce_sum(ba * bc, axis=-1)
            # TFLite-compatible arccos approximation
            x = tf.clip_by_value(cos_angle, -1.0, 1.0)
            return tf.constant(3.14159265359 / 2, dtype=tf.float32) - x - (x * x * x) / 6.0 - (x * x * x * x * x) / 40.0

        left_knee_angle = joint_angle(xy[:, 11], xy[:, 13], xy[:, 15])
        right_knee_angle = joint_angle(xy[:, 12], xy[:, 14], xy[:, 16])

        torso_vec = xy[:, 11] - xy[:, 5]  # hip - shoulder
        torso_vec = tf.math.l2_normalize(torso_vec, axis=-1)
        vertical = tf.constant([0.0, 1.0], dtype=tf.float32)
        vertical = tf.broadcast_to(vertical, tf.shape(torso_vec))
        back_alignment = tf.reduce_sum(torso_vec * vertical, axis=-1)

        knee_symmetry = xy[:, 13, 1] - xy[:, 14, 1]
        hip_knee_diff = (xy[:, 11, 1] + xy[:, 12, 1]) / 2.0 - (xy[:, 13, 1] + xy[:, 14, 1]) / 2.0

        feats = tf.stack([left_knee_angle, right_knee_angle, back_alignment, knee_symmetry, hip_knee_diff], axis=-1)

        # Tailor this for specific exercise
        return self.biomech_dense(feats)

    def generate_rule_based_outputs(self, keypoints):
        """
        Generate a single rule-based instruction and joint mask per sample.
        Returns:
            instructions: list of length batch_size (one string per sample)
            joint_mask: (batch, 17) 0/1 mask indicating joints to adjust
        """
        # Precompute engineered features once
        feats = self.compute_engineered_features(keypoints)  # (batch, 32)
        left_knee_angle = feats[:, 0]
        right_knee_angle = feats[:, 1]
        back_alignment = feats[:, 2]

        batch_size = tf.shape(keypoints)[0]
        joint_mask = tf.zeros((batch_size, self.num_joints), dtype=tf.float32)

        # Define rules: priority order matters
        rules = [
            (left_knee_angle < 160, [11, 13, 15], "knees_caving_in"),
            (right_knee_angle < 160, [12, 14, 16], "knees_caving_in"),
            (back_alignment < 0.8, [5, 11], "back_not_straight")
        ]

        # Initialize instruction list with empty strings
        instructions = [""] * batch_size

        # Apply rules in order; the first satisfied rule is taken
        for cond, joints, instr in rules:
            flag = tf.cast(cond, tf.float32)
            mask_update = tf.constant([1 if i in joints else 0 for i in range(self.num_joints)], dtype=tf.float32)
            joint_mask += tf.expand_dims(flag, axis=-1) * mask_update

            # Only set instruction if not already set
            for i in range(batch_size):
                if flag[i] > 0 and instructions[i] == "":
                    instructions[i] = instr

        joint_mask = tf.clip_by_value(joint_mask, 0, 1)
        return instructions, joint_mask

    def call(self, inputs, training=None):
        """
        Forward pass
        Inputs: (batch, 51) flattened keypoints [y, x, conf] * 17
        Outputs: dict with learned & rule-based results
        """
        # Reshape & preprocess
        x = self.reshape_layer(inputs)
        x_proc = self.preprocess_keypoints(x)

        # === CNN Backbone ===
        c = self.conv1(x_proc); c = self.bn1(c, training=training); c = self.relu1(c)
        c = self.conv2(c); c = self.bn2(c, training=training); c = self.relu2(c)
        c = self.conv3(c); c = self.bn3(c, training=training); c = self.relu3(c)
        c = self.global_pool(c)
        cnn_embedding = self.cnn_dense(c)

        # === Engineered Features ===
        biomech_embedding = self.compute_engineered_features(x_proc)

        # === Fusion ===
        fused = layers.Concatenate()([cnn_embedding, biomech_embedding])
        fused = self.fusion_dense1(fused); fused = self.fusion_dropout1(fused, training=training)
        fused = self.fusion_dense2(fused); fused = self.fusion_dropout2(fused, training=training)

        # === Learned Output ===
        form_score = self.form_score_output(fused)

        # === Rule-Based Outputs ===
        instructions, joint_mask = self.generate_rule_based_outputs(x_proc)

        # === Final 17-keypoint output with confidence channel as joint mask ===
        keypoints_out = tf.concat([x[:, :, :2], tf.expand_dims(joint_mask, axis=-1)], axis=-1)

        return {
            "form_score": form_score,
            "instructions": instructions,
            "joint_mask": joint_mask,
            "keypoints_out": keypoints_out  # 17×3 array with confidence as correction mask
        }