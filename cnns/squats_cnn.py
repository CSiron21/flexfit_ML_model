import tensorflow as tf
from tensorflow.keras import layers

# Instruction decoding for external use
INSTRUCTION_MAP = {
    0: "good_form",
    1: "knee_forward_over_ankle",
    2: "shoulder_forward",
}

class SquatPoseCNN(tf.keras.Model):
	"""
	Squat-focused 1D CNN with engineered biomechanical features and rule-based outputs.
	Optimized for sideways camera views where the hidden leg cannot be reliably observed.

	Inputs: (batch, 51) flattened keypoints [y, x, conf] * 17
	Outputs:
	- form_score: (batch, 1) sigmoid in [0,1]
	- instruction_id: (batch,) int32 as per INSTRUCTION_MAP
	- joint_masked_keypoints: (batch, 17, 3) with confidence channel as binary correction mask
	"""

	def __init__(
		self,
		num_joints: int = 17,
		knee_forward_threshold: float = 0.30,
		knee_forward_tolerance: float = 0.05,
		shoulder_forward_threshold: float = 0.18,
		shoulder_forward_tolerance: float = 0.05,
		**kwargs,
	):
		super(SquatPoseCNN, self).__init__(**kwargs)
		self.num_joints = num_joints
		self.knee_forward_threshold = knee_forward_threshold
		self.knee_forward_tolerance = knee_forward_tolerance
		self.shoulder_forward_threshold = shoulder_forward_threshold
		self.shoulder_forward_tolerance = shoulder_forward_tolerance
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

		self.fusion_dense1 = layers.Dense(48, activation='relu')
		self.fusion_dropout1 = layers.Dropout(0.3)
		self.fusion_dense2 = layers.Dense(24, activation='relu')
		self.fusion_dropout2 = layers.Dropout(0.2)
		self.form_score_output = layers.Dense(1, activation='sigmoid', name='form_score')

	def preprocess_keypoints(self, keypoints: tf.Tensor) -> tf.Tensor:
		"""
		TFLite-optimized preprocessing:
		- Center by mid-hip
		- Scale by torso height
		- Apply preprocessing joint mask efficiently
		Input: (batch, 17, 3) → Output: (batch, 17, 2) - coordinates only for CNN
		"""
		xy = keypoints[:, :, :2]
		conf = keypoints[:, :, 2:3]

		# Joint mask: keep shoulders, hips, knees, ankles
		preproc_mask = tf.constant([
			0, 0, 0,  # nose, eyes
			0, 0,     # ears
			1, 1,     # shoulders
			0, 0,     # elbows
			0, 0,     # wrists
			1, 1,     # hips
			1, 1,     # knees
			1, 1,     # ankles
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

		# Return only coordinates for CNN (no confidence channel)
		processed = scaled * mask[:, :, :2]

		return processed

	def _compute_knee_angle_and_direction(self, hip: tf.Tensor, knee: tf.Tensor, ankle: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		"""
		Compute the interior angle and bend direction at the knee joint (hip-knee-ankle).
		Uses radians internally for TFLite compatibility.
		
		Args:
			hip: (batch, 2) hip coordinates [y, x]
			knee: (batch, 2) knee coordinates [y, x] 
			ankle: (batch, 2) ankle coordinates [y, x]
			
		Returns:
			tuple of:
			- (batch,) interior angle in radians
			- (batch,) bend direction: +1 for one side, -1 for other side
		"""
		# Vectors from knee to hip and knee to ankle
		vec1 = hip - knee  # knee to hip
		vec2 = ankle - knee  # knee to ankle
		
		# Compute dot product and magnitudes for angle
		dot_product = tf.reduce_sum(vec1 * vec2, axis=1)
		mag1 = tf.norm(vec1, axis=1)
		mag2 = tf.norm(vec2, axis=1)
		
		# Avoid division by zero
		magnitude_product = tf.maximum(mag1 * mag2, 1e-8)
		
		# Compute cosine of angle and clamp to valid range [-1, 1]
		cos_angle = tf.clip_by_value(dot_product / magnitude_product, -1.0, 1.0)
		
		# TFLite-compatible arccos approximation using polynomial
		x = cos_angle
		angle_rad = tf.constant(3.14159265359 / 2, dtype=tf.float32) - x - (x * x * x) / 6.0 - (x * x * x * x * x) / 40.0
		
		# Compute cross product for bend direction (2D cross product)
		# cross = vec1_x * vec2_y - vec1_y * vec2_x
		cross_product = vec1[:, 1] * vec2[:, 0] - vec1[:, 0] * vec2[:, 1]
		
		# Use tf.where to ensure we never get 0 direction
		bend_direction = tf.where(cross_product >= 0, 1.0, -1.0)
		
		return angle_rad, bend_direction

	def _raw_engineered_features(self, keypoints: tf.Tensor) -> tf.Tensor:
		"""
		Compute and normalize squat-relevant features for sideways analysis.
		Returns tensor (batch, 2) with features:
		[knee_forward_offset_normalized, shoulder_over_knee_normalized]
		All operations are vectorized and TFLite-friendly.
		"""
		yx = keypoints[:, :, :2]
		conf = keypoints[:, :, 2]

		# Select side based on higher knee confidence
		left_knee_conf = conf[:, 13]
		right_knee_conf = conf[:, 14]
		is_left = tf.cast(left_knee_conf > right_knee_conf, tf.float32)
		is_right = 1.0 - is_left

		# Helper to select same-side joint consistently
		def pick(side_left_idx: int, side_right_idx: int) -> tf.Tensor:
			return (yx[:, side_left_idx, :] * is_left[:, None] + 
					yx[:, side_right_idx, :] * is_right[:, None])

		# Same-side triplet (hip, knee, ankle)
		hip = pick(11, 12)
		knee = pick(13, 14)
		ankle = pick(15, 16)

		# Knee-forward offset normalization
		knee_x = knee[:, 1]
		ankle_x = ankle[:, 1]
		knee_forward_offset = tf.abs(knee_x - ankle_x)
		knee_thresh = tf.convert_to_tensor(self.knee_forward_threshold + self.knee_forward_tolerance, dtype=knee_forward_offset.dtype)
		knee_thresh = tf.maximum(knee_thresh, tf.constant(1e-6, dtype=knee_forward_offset.dtype))
		knee_excess = tf.maximum(0.0, knee_forward_offset - knee_thresh)
		knee_forward_norm = knee_excess / knee_thresh
		knee_forward_norm = tf.clip_by_value(knee_forward_norm, 0.0, 1.0)

		# Orientation-aware shoulder over knee forward offset
		shoulder = pick(5, 6)
		shoulder_x = shoulder[:, 1]
		
		# Compute knee angle and bend direction to determine if squatting and facing direction
		knee_angle_rad, bend_direction = self._compute_knee_angle_and_direction(hip, knee, ankle)
		
		# Check if squatting (angle < 130 degrees = ~2.27 radians)
		is_squatting = knee_angle_rad < 2.27  # Keep as boolean for tf.where
		
		# Determine facing direction based on actual knee bend direction
		facing_left_bool = bend_direction > 0  # Keep as boolean for tf.where
		facing_right_bool = bend_direction < 0  # Keep as boolean for tf.where
		facing_left = tf.cast(facing_left_bool, tf.float32)  # Cast to float for arithmetic
		facing_right = tf.cast(facing_right_bool, tf.float32)  # Cast to float for arithmetic
		
		# Compute shoulder offset based on facing direction
		shoulder_offset_left = knee_x - shoulder_x  # positive if shoulder behind knee
		shoulder_offset_right = shoulder_x - knee_x  # positive if shoulder behind knee
		
		# Select offset based on facing direction
		shoulder_offset = shoulder_offset_left * facing_left + shoulder_offset_right * facing_right
		
		# Only check shoulder position if squatting
		shoulder_over_knee_offset = tf.where(
			is_squatting,
			tf.maximum(0.0, -shoulder_offset),  # negative offset means shoulder too far forward
			tf.zeros_like(shoulder_offset)
		)
		
		shoulder_thresh = tf.convert_to_tensor(self.shoulder_forward_threshold + self.shoulder_forward_tolerance, dtype=shoulder_over_knee_offset.dtype)
		shoulder_thresh = tf.maximum(shoulder_thresh, tf.constant(1e-6, dtype=shoulder_over_knee_offset.dtype))
		shoulder_excess = tf.maximum(0.0, shoulder_over_knee_offset - shoulder_thresh)
		shoulder_over_knee_norm = shoulder_excess / shoulder_thresh
		shoulder_over_knee_norm = tf.clip_by_value(shoulder_over_knee_norm, 0.0, 1.0)

		return tf.stack([knee_forward_norm, shoulder_over_knee_norm], axis=1)

	def compute_engineered_features(self, keypoints: tf.Tensor) -> tf.Tensor:
		normalized_feats = self._raw_engineered_features(keypoints)
		return self.biomech_dense(normalized_feats)

	def _rule_outputs(self, keypoints: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		"""
		TFLite-optimized vectorized rule-based outputs for squat instructions.
		
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

		# Select side based on higher knee confidence
		left_knee_conf = conf[:, 13]
		right_knee_conf = conf[:, 14]
		is_left = tf.cast(left_knee_conf > right_knee_conf, tf.float32)
		is_right = 1.0 - is_left

		# Helper to select same-side joint consistently
		def pick(side_left_idx: int, side_right_idx: int) -> tf.Tensor:
			return (yx[:, side_left_idx, :] * is_left[:, None] + 
					yx[:, side_right_idx, :] * is_right[:, None])

		# Same-side triplet (hip, knee, ankle)
		hip = pick(11, 12)
		knee = pick(13, 14)
		ankle = pick(15, 16)

		# Knee-forward check
		knee_x = knee[:, 1]
		ankle_x = ankle[:, 1]
		knee_forward_offset = tf.abs(knee_x - ankle_x)
		cond_knee_forward = knee_forward_offset > (self.knee_forward_threshold + self.knee_forward_tolerance)
		knee_forward_deviation = tf.maximum(0.0, knee_forward_offset - (self.knee_forward_threshold + self.knee_forward_tolerance))

		# Orientation-aware shoulder-forward check
		shoulder = pick(5, 6)
		shoulder_x = shoulder[:, 1]
		
		# Compute knee angle and bend direction to determine if squatting and facing direction
		knee_angle_rad, bend_direction = self._compute_knee_angle_and_direction(hip, knee, ankle)
		
		# Check if squatting (angle < 130 degrees = ~2.27 radians)
		is_squatting = knee_angle_rad < 2.27  # Keep as boolean for tf.where
		
		# Determine facing direction based on actual knee bend direction
		facing_left_bool = bend_direction > 0 
		facing_right_bool = bend_direction < 0  
		facing_left = tf.cast(facing_left_bool, tf.float32) 
		facing_right = tf.cast(facing_right_bool, tf.float32) 
		
		# Compute shoulder offset based on facing direction
		shoulder_offset_left = knee_x - shoulder_x  # positive if shoulder behind knee
		shoulder_offset_right = shoulder_x - knee_x  # positive if shoulder behind knee
		
		# Select offset based on facing direction
		shoulder_offset = shoulder_offset_left * facing_left + shoulder_offset_right * facing_right
		
		# Only check shoulder position if squatting
		shoulder_forward_offset = tf.where(
			is_squatting,
			tf.maximum(0.0, -shoulder_offset),  # negative offset means shoulder too far forward
			tf.zeros_like(shoulder_offset)
		)
		
		cond_shoulder_forward = shoulder_forward_offset > (self.shoulder_forward_threshold + self.shoulder_forward_tolerance)
		shoulder_forward_deviation = tf.maximum(0.0, shoulder_forward_offset - (self.shoulder_forward_threshold + self.shoulder_forward_tolerance))

		# Find rule with maximum deviation (most critical issue)
		deviation_stack = tf.stack([knee_forward_deviation, shoulder_forward_deviation], axis=1)
		max_deviation_idx = tf.argmax(deviation_stack, axis=1)
		max_deviation = tf.reduce_max(deviation_stack, axis=1)
		
		any_issue = max_deviation > 0.0
		instruction_id = tf.where(any_issue, max_deviation_idx + 1, tf.zeros_like(max_deviation_idx))

		# Joint mask for selected instruction
		one_hot = tf.one_hot(max_deviation_idx, depth=2, dtype=tf.float32)
		
		mask_knee_forward = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], dtype=tf.float32)
		mask_shoulder_forward = tf.constant([
			0,0,0,0,0,  # 0-4
			1,1,        # 5-6 shoulders
			0,0,        # 7-8 elbows
			0,0,        # 9-10 wrists
			1,1,        # 11-12 hips
			1,1,        # 13-14 knees
			0,0         # 15-16 ankles
		], dtype=tf.float32)
		
		masks = tf.stack([mask_knee_forward, mask_shoulder_forward], axis=0)
		joint_mask = tf.tensordot(one_hot, masks, axes=[[1],[0]])
		joint_mask = joint_mask * tf.cast(any_issue, tf.float32)[:, None]
		joint_mask = tf.clip_by_value(joint_mask, 0.0, 1.0)

		return instruction_id, joint_mask

	def _create_normalized_keypoints_with_confidence(self, keypoints: tf.Tensor) -> tf.Tensor:
		"""
		Creates normalized keypoints (coordinates) with confidence channel.
		Input: (batch, 17, 3) - RAW keypoints [y, x, conf]
		Output: (batch, 17, 3) - Normalized coordinates [y, x, conf]
		"""
		xy = keypoints[:, :, :2]
		conf = keypoints[:, :, 2:3]

		# Joint mask: keep shoulders, hips, knees, ankles
		preproc_mask = tf.constant([
			0, 0, 0,  # nose, eyes
			0, 0,     # ears
			1, 1,     # shoulders
			0, 0,     # elbows
			0, 0,     # wrists
			1, 1,     # hips
			1, 1,     # knees
			1, 1,     # ankles
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

		# Return normalized coordinates with confidence channel
		normalized_keypoints = scaled * mask[:, :, :2]
		normalized_keypoints = tf.concat([normalized_keypoints, conf], axis=-1)

		return normalized_keypoints

	def call(self, inputs: tf.Tensor, training=None):
		if inputs.shape[-1] != 51:
			raise ValueError(f"Expected input shape (..., 51), got {inputs.shape}")
		
		x = self.reshape_layer(inputs)  # (batch, 17, 3) - RAW keypoints with confidence
		x_proc = self.preprocess_keypoints(x)  # (batch, 17, 2) - Coordinates only for CNN
		
		# Create normalized keypoints with confidence for biomechanical features and rules
		x_normalized = self._create_normalized_keypoints_with_confidence(x)  # (batch, 17, 3) - Normalized + confidence

		# CNN path: uses preprocessed coordinates only
		c = self.conv1(x_proc); c = self.bn1(c, training=training); c = self.relu1(c)
		c = self.conv2(c); c = self.bn2(c, training=training); c = self.relu2(c)
		c = self.conv3(c); c = self.bn3(c, training=training); c = self.relu3(c)
		c = self.global_pool(c)
		cnn_embedding = self.cnn_dense(c)

		# Biomechanical features: uses NORMALIZED keypoints (with confidence for side selection)
		biomech_embedding = self.compute_engineered_features(x_normalized)

		fused = layers.Concatenate()([cnn_embedding, biomech_embedding])
		fused = self.fusion_dense1(fused); fused = self.fusion_dropout1(fused, training=training)
		fused = self.fusion_dense2(fused); fused = self.fusion_dropout2(fused, training=training)

		cnn_form_score = self.form_score_output(fused)
		
		# Rule outputs: uses NORMALIZED keypoints (with confidence for side selection)
		instruction_id, joint_mask = self._rule_outputs(x_normalized)
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
		"""Internal build method for SquatPoseCNN model."""
		model = SquatPoseCNN()
		
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
	return SquatPoseCNN._build_model_internal(dropout_rate=dropout_rate, learning_rate=learning_rate)
