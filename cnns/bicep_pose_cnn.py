import tensorflow as tf
from tensorflow.keras import layers

# Instruction decoding for external use
INSTRUCTION_MAP = {
    0: "good_form",
    1: "shoulder_elbow_misaligned",
    2: "shoulder_hip_misaligned",
}

class BicepPoseCNN(tf.keras.Model):
	"""
	Bicep curl-focused 1D CNN with engineered biomechanical features and rule-based outputs.
	Optimized for sideways camera views where the hidden arm cannot be reliably observed.
	
	Shoulder-Elbow Rule: Single-unit analysis using confidence-based side selection:
	- Selects the shoulder-elbow pair with higher confidence score from Movenet for measurement
	- Computes shoulder_elbow_offset = |shoulder_x - elbow_x| for the selected side only
	- Triggers if offset exceeds shoulder_elbow_threshold + tolerance (5% wiggle room)
	- Benefits: robust to occlusion, sideways views, prevents false positives from hidden arm
	- Visual feedback highlights ONLY the shoulder and elbow of the higher-confidence side
	
	Shoulder-Hip Rule: Ensures torso stays upright during curl:
	- Uses same-side shoulder and hip for consistent measurement
	- Computes shoulder_hip_offset = |shoulder_x - hip_x| for the selected side only
	- Triggers if offset exceeds shoulder_hip_threshold + tolerance
	- Visual feedback highlights ONLY the shoulder and hip of the higher-confidence side

	Features: Focuses on robust, visible-side measurements only:
	- shoulder_elbow_offset: Single-unit horizontal offset using higher-confidence side only
	- shoulder_hip_offset: Torso alignment offset using same-side joints
	- Both features use consistent same-side joint selection for reliability

	Inputs: (batch, 51) flattened keypoints [y, x, conf] * 17
	Outputs:
	- form_score: (batch, 1) sigmoid in [0,1]
	- instruction_id: (batch,) int32 as per INSTRUCTION_MAP
	- joint_masked_keypoints: (batch, 17, 3) with confidence channel as binary correction mask
	"""

	def __init__(
		self,
		num_joints: int = 17,
		shoulder_elbow_threshold: float = 0.15,
		shoulder_elbow_tolerance: float = 0.05,
		shoulder_hip_threshold: float = 0.20,
		shoulder_hip_tolerance: float = 0.05,
		**kwargs,
	):
		super(BicepPoseCNN, self).__init__(**kwargs)
		self.num_joints = num_joints
		self.shoulder_elbow_threshold = shoulder_elbow_threshold
		self.shoulder_elbow_tolerance = shoulder_elbow_tolerance
		self.shoulder_hip_threshold = shoulder_hip_threshold
		self.shoulder_hip_tolerance = shoulder_hip_tolerance
		self.reshape_layer = layers.Reshape((num_joints, 3))

		# CNN Backbone
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

		self.biomech_dense = layers.Dense(32, activation='relu', name="biomech_embedding")

		self.fusion_dense1 = layers.Dense(48, activation='relu')  # Reduced from 64
		self.fusion_dropout1 = layers.Dropout(0.3)
		self.fusion_dense2 = layers.Dense(24, activation='relu')  # Reduced from 32
		self.fusion_dropout2 = layers.Dropout(0.2)
		self.form_score_output = layers.Dense(1, activation='sigmoid', name='form_score')

	def preprocess_keypoints(self, keypoints: tf.Tensor) -> tf.Tensor:
		"""
		TFLite-optimized preprocessing:
		- Center by mid-hip
		- Scale by torso height
		- Apply preprocessing joint mask efficiently
		Input: (batch, 17, 3) → Output: (batch, 17, 3)
		"""
		xy = keypoints[:, :, :2]
		conf = keypoints[:, :, 2:3]

		# Joint mask: keep shoulders, elbows, wrists, hips
		preproc_mask = tf.constant([
			0, 0, 0,  # nose, eyes
			0, 0,     # ears
			1, 1,     # shoulders
			1, 1,     # elbows
			1, 1,     # wrists
			1, 1,     # hips
			0, 0,     # knees
			0, 0,     # ankles
		], dtype=tf.float32)
		
		mask = tf.reshape(preproc_mask, (1, self.num_joints, 1))
		
		hip_slice = xy[:, 11:13, :] * mask[:, 11:13, :]
		shoulder_slice = xy[:, 5:7, :] * mask[:, 5:7, :]

		mid_hip = tf.reduce_sum(hip_slice, axis=1, keepdims=True) / 2.0
		mid_shoulder = tf.reduce_sum(shoulder_slice, axis=1, keepdims=True) / 2.0

		centered = xy - mid_hip
		torso_vec = mid_shoulder - mid_hip
		torso_height = tf.norm(torso_vec, axis=2, keepdims=True)
		torso_height = tf.maximum(torso_height, 1e-6)
		scaled = centered / torso_height

		processed = tf.concat([scaled, conf], axis=2) * mask

		return processed

	def _raw_engineered_features(self, keypoints: tf.Tensor) -> tf.Tensor:
		"""
		Compute and normalize bicep curl-relevant features for sideways analysis.
		Returns tensor (batch, 2) with features:
		[shoulder_elbow_offset_normalized, shoulder_hip_offset_normalized]
		- shoulder_elbow_offset_normalized in [0,1], 0 small/none, 1 at/above threshold
		- shoulder_hip_offset_normalized in [0,1], 0 small/none, 1 at/above threshold
		All operations are vectorized and TFLite-friendly.
		"""
		yx = keypoints[:, :, :2]
		conf = keypoints[:, :, 2]

		# Select side with higher shoulder confidence
		shoulder_conf = tf.stack([conf[:, 5], conf[:, 6]], axis=1)
		is_left = tf.cast(shoulder_conf[:, 0] > shoulder_conf[:, 1], tf.float32)
		is_right = 1.0 - is_left

		# Shoulder-elbow offset normalization
		shoulder_x = tf.stack([yx[:, 5, 1], yx[:, 6, 1]], axis=1)
		elbow_x = tf.stack([yx[:, 7, 1], yx[:, 8, 1]], axis=1)
		shoulder_elbow_offsets = tf.abs(shoulder_x - elbow_x)
		shoulder_elbow_offset = shoulder_elbow_offsets[:, 0] * is_left + shoulder_elbow_offsets[:, 1] * is_right
		shoulder_elbow_thresh = tf.convert_to_tensor(self.shoulder_elbow_threshold + self.shoulder_elbow_tolerance, dtype=shoulder_elbow_offset.dtype)
		shoulder_elbow_thresh = tf.maximum(shoulder_elbow_thresh, tf.constant(1e-6, dtype=shoulder_elbow_offset.dtype))
		shoulder_elbow_excess = tf.maximum(0.0, shoulder_elbow_offset - shoulder_elbow_thresh)
		shoulder_elbow_norm = shoulder_elbow_excess / shoulder_elbow_thresh
		shoulder_elbow_norm = tf.clip_by_value(shoulder_elbow_norm, 0.0, 1.0)

		# Shoulder-hip offset normalization (same-side selection)
		shoulder = (yx[:, 5, :] * is_left[:, None] + yx[:, 6, :] * is_right[:, None])
		hip = (yx[:, 11, :] * is_left[:, None] + yx[:, 12, :] * is_right[:, None])
		shoulder_x_same = shoulder[:, 1]
		hip_x_same = hip[:, 1]
		shoulder_hip_offset = tf.abs(shoulder_x_same - hip_x_same)
		shoulder_hip_thresh = tf.convert_to_tensor(self.shoulder_hip_threshold + self.shoulder_hip_tolerance, dtype=shoulder_hip_offset.dtype)
		shoulder_hip_thresh = tf.maximum(shoulder_hip_thresh, tf.constant(1e-6, dtype=shoulder_hip_offset.dtype))
		shoulder_hip_excess = tf.maximum(0.0, shoulder_hip_offset - shoulder_hip_thresh)
		shoulder_hip_norm = shoulder_hip_excess / shoulder_hip_thresh
		shoulder_hip_norm = tf.clip_by_value(shoulder_hip_norm, 0.0, 1.0)

		return tf.stack([shoulder_elbow_norm, shoulder_hip_norm], axis=1)

	def compute_engineered_features(self, keypoints: tf.Tensor) -> tf.Tensor:
		normalized_feats = self._raw_engineered_features(keypoints)
		return self.biomech_dense(normalized_feats)

	def _rule_outputs(self, keypoints: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		"""
		TFLite-optimized vectorized rule-based outputs for bicep curl instructions.
		
		Rule Selection Logic:
		- Instead of picking the first triggered rule, selects the rule with MAXIMUM deviation
		- This prioritizes the most critical form issue for user feedback
		- Deviation = how much the measurement exceeds the threshold (0.0 if within limits)
		
		Returns:
		- instruction_id: (batch,) int32 - Rule with maximum deviation (0 = good form)
		- joint_mask: (batch, 17) float32 in {0,1} - Joints to highlight for the selected rule
		"""
		yx = keypoints[:, :, :2]
		conf = keypoints[:, :, 2]

		# Select side based on higher shoulder confidence
		left_shoulder_conf = conf[:, 5]
		right_shoulder_conf = conf[:, 6]
		is_left = tf.cast(left_shoulder_conf > right_shoulder_conf, tf.float32)
		is_right = 1.0 - is_left

		# Helper to select same-side joint consistently
		def pick(side_left_idx: int, side_right_idx: int) -> tf.Tensor:
			return (yx[:, side_left_idx, :] * is_left[:, None] + 
					yx[:, side_right_idx, :] * is_right[:, None])

		# Same-side joints
		shoulder = pick(5, 6)
		elbow = pick(7, 8)
		hip = pick(11, 12)

		# Shoulder-elbow check
		shoulder_x = shoulder[:, 1]
		elbow_x = elbow[:, 1]
		shoulder_elbow_offset = tf.abs(shoulder_x - elbow_x)
		cond_shoulder_elbow = shoulder_elbow_offset > (self.shoulder_elbow_threshold + self.shoulder_elbow_tolerance)
		shoulder_elbow_deviation = tf.maximum(0.0, shoulder_elbow_offset - (self.shoulder_elbow_threshold + self.shoulder_elbow_tolerance))

		# Shoulder-hip check
		shoulder_x_same = shoulder[:, 1]
		hip_x = hip[:, 1]
		shoulder_hip_offset = tf.abs(shoulder_x_same - hip_x)
		cond_shoulder_hip = shoulder_hip_offset > (self.shoulder_hip_threshold + self.shoulder_hip_tolerance)
		shoulder_hip_deviation = tf.maximum(0.0, shoulder_hip_offset - (self.shoulder_hip_threshold + self.shoulder_hip_tolerance))

		# Find rule with maximum deviation (most critical issue)
		deviation_stack = tf.stack([shoulder_elbow_deviation, shoulder_hip_deviation], axis=1)
		max_deviation_idx = tf.argmax(deviation_stack, axis=1)
		max_deviation = tf.reduce_max(deviation_stack, axis=1)
		
		any_issue = max_deviation > 0.0
		instruction_id = tf.where(any_issue, max_deviation_idx + 1, tf.zeros_like(max_deviation_idx))

		# Joint mask for selected instruction
		one_hot = tf.one_hot(max_deviation_idx, depth=2, dtype=tf.float32)
		
		mask_shoulder_elbow = tf.constant([
			0,0,0,0,0,  # 0-4
			1,1,        # 5-6 shoulders
			1,1,        # 7-8 elbows
			0,0,        # 9-10 wrists
			0,0,        # 11-12 hips
			0,0,        # 13-14 knees
			0,0         # 15-16 ankles
		], dtype=tf.float32)
		
		mask_shoulder_hip = tf.constant([
			0,0,0,0,0,  # 0-4
			1,1,        # 5-6 shoulders
			0,0,        # 7-8 elbows
			0,0,        # 9-10 wrists
			1,1,        # 11-12 hips
			0,0,        # 13-14 knees
			0,0         # 15-16 ankles
		], dtype=tf.float32)
		
		masks = tf.stack([mask_shoulder_elbow, mask_shoulder_hip], axis=0)
		joint_mask = tf.tensordot(one_hot, masks, axes=[[1],[0]])
		joint_mask = joint_mask * tf.cast(any_issue, tf.float32)[:, None]
		joint_mask = tf.clip_by_value(joint_mask, 0.0, 1.0)

		return instruction_id, joint_mask

	def call(self, inputs: tf.Tensor, training=None):
		if inputs.shape[-1] != 51:
			raise ValueError(f"Expected input shape (..., 51), got {inputs.shape}")
		
		x = self.reshape_layer(inputs)
		x_proc = self.preprocess_keypoints(x)

		c = self.conv1(x_proc); c = self.bn1(c, training=training); c = self.relu1(c)
		c = self.conv2(c); c = self.bn2(c, training=training); c = self.relu2(c)
		c = self.conv3(c); c = self.bn3(c, training=training); c = self.relu3(c)
		c = self.global_pool(c)
		cnn_embedding = self.cnn_dense(c)

		biomech_embedding = self.compute_engineered_features(x_proc)

		fused = layers.Concatenate()([cnn_embedding, biomech_embedding])
		fused = self.fusion_dense1(fused); fused = self.fusion_dropout1(fused, training=training)
		fused = self.fusion_dense2(fused); fused = self.fusion_dropout2(fused, training=training)

		cnn_form_score = self.form_score_output(fused)
		
		instruction_id, joint_mask = self._rule_outputs(x_proc)
		instruction_id = tf.cast(instruction_id, tf.int32)
		
		if training:
			final_form_score = cnn_form_score
		else:
			rule_confidence = tf.where(
				instruction_id == 0,
				tf.ones_like(cnn_form_score) * 0.95,
				tf.ones_like(cnn_form_score) * 0.50
			)

			final_form_score = tf.where(
				instruction_id == 0,
				cnn_form_score * 0.3 + rule_confidence * 0.7, 
				cnn_form_score * 0.4 + rule_confidence * 0.6,
			)
		
		joint_masked_keypoints = tf.concat([x[:, :, :2], tf.expand_dims(joint_mask, axis=-1)], axis=-1)

		return {
			'form_score': final_form_score,
			'instruction_id': instruction_id,
			'joint_masked_keypoints': joint_masked_keypoints
		}

	@staticmethod
	def _build_model_internal(dropout_rate: float = 0.4, learning_rate: float = 0.0005) -> tf.keras.Model:
		"""Internal build method for BicepPoseCNN model."""
		model = BicepPoseCNN()
		
		model.fusion_dropout1.rate = dropout_rate
		model.fusion_dropout2.rate = dropout_rate * 0.67
		
		losses = {
			'form_score': 'binary_crossentropy',
			'instruction_id': 'mse',
			'joint_masked_keypoints': 'mse'
		}
		
		loss_weights = {
			'form_score': 1.0,
			'instruction_id': 0.0,
			'joint_masked_keypoints': 0.0
		}
		
		metrics = {
			'form_score': ['accuracy'],
			'instruction_id': [],
			'joint_masked_keypoints': []
		}
		
		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
			loss=losses,
			loss_weights=loss_weights,
			metrics=metrics
		)
		
		return model


def build_model(dropout_rate: float = 0.4, learning_rate: float = 0.0005) -> tf.keras.Model:
	"""Module-level build_model function for easy importing by train.py."""
	return BicepPoseCNN._build_model_internal(dropout_rate=dropout_rate, learning_rate=learning_rate)
