import tensorflow as tf
from tensorflow.keras import layers

# Instruction decoding for external use
INSTRUCTION_MAP = {
    0: "good_form",
    1: "elbow_flare",
    2: "wrist_asymmetry",
}

class OverheadPoseCNN(tf.keras.Model):
	"""
	Overhead Press-focused 1D CNN with engineered biomechanical features and rule-based outputs.
	Optimized for front-view camera analysis where both sides are independently observable.
	
	Elbow Flare Rule: Ensures elbows do not flare outward during press:
	- Computes horizontal (x-axis) offset between wrist and elbow for each side independently
	- Triggers if offset exceeds elbow_flare_threshold + tolerance
	- Visual feedback highlights wrist and elbow on the affected side(s)
	
	Wrist Symmetry Rule: Ensures both wrists rise evenly at the same height:
	- Computes vertical (y-axis) difference between left and right wrists
	- Triggers if difference exceeds wrist_symmetry_threshold + tolerance
	- Visual feedback highlights both wrists and elbows
	
	Features: Focuses on front-view measurements:
	- elbow_flare_offset: Horizontal misalignment between wrist and elbow (both sides)
	- wrist_symmetry_offset: Vertical difference between left and right wrists
	- Both features use independent side analysis for front-view reliability

	Inputs: (batch, 51) flattened keypoints [y, x, conf] * 17
	Outputs:
	- form_score: (batch, 1) sigmoid in [0,1]
	- instruction_id: (batch,) int32 as per INSTRUCTION_MAP
	- joint_masked_keypoints: (batch, 17, 3) with confidence channel as binary correction mask
	"""

	def __init__(
		self,
		num_joints: int = 17,
		elbow_flare_threshold: float = 0.15,
		elbow_flare_tolerance: float = 0.05,
		wrist_symmetry_threshold: float = 0.12,
		wrist_symmetry_tolerance: float = 0.05,
		**kwargs,
	):
		super(OverheadPoseCNN, self).__init__(**kwargs)
		self.num_joints = num_joints
		self.elbow_flare_threshold = elbow_flare_threshold
		self.elbow_flare_tolerance = elbow_flare_tolerance
		self.wrist_symmetry_threshold = wrist_symmetry_threshold
		self.wrist_symmetry_tolerance = wrist_symmetry_tolerance
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
		Compute and normalize overhead press-relevant features for front-view analysis.
		Returns tensor (batch, 2) with features:
		[elbow_flare_offset_normalized, wrist_symmetry_offset_normalized]
		- elbow_flare_offset_normalized in [0,1], 0 small/none, 1 at/above threshold
		- wrist_symmetry_offset_normalized in [0,1], 0 small/none, 1 at/above threshold
		All operations are vectorized and TFLite-friendly.
		"""
		yx = keypoints[:, :, :2]
		conf = keypoints[:, :, 2]

		# Elbow flare check: horizontal offset between wrist and elbow for each side
		left_wrist_x = yx[:, 9, 1]   # Left wrist x-coordinate
		left_elbow_x = yx[:, 7, 1]   # Left elbow x-coordinate
		right_wrist_x = yx[:, 10, 1] # Right wrist x-coordinate
		right_elbow_x = yx[:, 8, 1]  # Right elbow x-coordinate

		left_flare_offset = tf.abs(left_wrist_x - left_elbow_x)
		right_flare_offset = tf.abs(right_wrist_x - right_elbow_x)
		
		# Use maximum flare offset from either side
		max_flare_offset = tf.maximum(left_flare_offset, right_flare_offset)
		
		elbow_thresh = tf.convert_to_tensor(self.elbow_flare_threshold + self.elbow_flare_tolerance, dtype=max_flare_offset.dtype)
		elbow_thresh = tf.maximum(elbow_thresh, tf.constant(1e-6, dtype=max_flare_offset.dtype))
		elbow_excess = tf.maximum(0.0, max_flare_offset - elbow_thresh)
		elbow_flare_norm = elbow_excess / elbow_thresh
		elbow_flare_norm = tf.clip_by_value(elbow_flare_norm, 0.0, 1.0)

		# Wrist symmetry check: vertical difference between left and right wrists
		left_wrist_y = yx[:, 9, 0]   # Left wrist y-coordinate
		right_wrist_y = yx[:, 10, 0] # Right wrist y-coordinate
		
		wrist_symmetry_offset = tf.abs(left_wrist_y - right_wrist_y)
		
		wrist_thresh = tf.convert_to_tensor(self.wrist_symmetry_threshold + self.wrist_symmetry_tolerance, dtype=wrist_symmetry_offset.dtype)
		wrist_thresh = tf.maximum(wrist_thresh, tf.constant(1e-6, dtype=wrist_symmetry_offset.dtype))
		wrist_excess = tf.maximum(0.0, wrist_symmetry_offset - wrist_thresh)
		wrist_symmetry_norm = wrist_excess / wrist_thresh
		wrist_symmetry_norm = tf.clip_by_value(wrist_symmetry_norm, 0.0, 1.0)

		return tf.stack([elbow_flare_norm, wrist_symmetry_norm], axis=1)

	def compute_engineered_features(self, keypoints: tf.Tensor) -> tf.Tensor:
		normalized_feats = self._raw_engineered_features(keypoints)
		return self.biomech_dense(normalized_feats)

	def _rule_outputs(self, keypoints: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		"""
		TFLite-optimized vectorized rule-based outputs for overhead press instructions.
		
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

		# Elbow flare check: horizontal offset between wrist and elbow for each side
		left_wrist_x = yx[:, 9, 1]   # Left wrist x-coordinate
		left_elbow_x = yx[:, 7, 1]   # Left elbow x-coordinate
		right_wrist_x = yx[:, 10, 1] # Right wrist x-coordinate
		right_elbow_x = yx[:, 8, 1]  # Right elbow x-coordinate

		left_flare_offset = tf.abs(left_wrist_x - left_elbow_x)
		right_flare_offset = tf.abs(right_wrist_x - right_elbow_x)
		max_flare_offset = tf.maximum(left_flare_offset, right_flare_offset)
		
		cond_elbow_flare = max_flare_offset > (self.elbow_flare_threshold + self.elbow_flare_tolerance)
		elbow_flare_deviation = tf.maximum(0.0, max_flare_offset - (self.elbow_flare_threshold + self.elbow_flare_tolerance))

		# Wrist symmetry check: vertical difference between left and right wrists
		left_wrist_y = yx[:, 9, 0]   # Left wrist y-coordinate
		right_wrist_y = yx[:, 10, 0] # Right wrist y-coordinate
		
		wrist_symmetry_offset = tf.abs(left_wrist_y - right_wrist_y)
		cond_wrist_symmetry = wrist_symmetry_offset > (self.wrist_symmetry_threshold + self.wrist_symmetry_tolerance)
		wrist_symmetry_deviation = tf.maximum(0.0, wrist_symmetry_offset - (self.wrist_symmetry_threshold + self.wrist_symmetry_tolerance))

		# Find rule with maximum deviation (most critical issue)
		deviation_stack = tf.stack([elbow_flare_deviation, wrist_symmetry_deviation], axis=1)
		max_deviation_idx = tf.argmax(deviation_stack, axis=1)
		max_deviation = tf.reduce_max(deviation_stack, axis=1)
		
		any_issue = max_deviation > 0.0
		instruction_id = tf.where(any_issue, max_deviation_idx + 1, tf.zeros_like(max_deviation_idx))

		# Joint mask for selected instruction
		one_hot = tf.one_hot(max_deviation_idx, depth=2, dtype=tf.float32)
		
		# Mask for elbow flare: highlight wrists and elbows
		mask_elbow_flare = tf.constant([
			0,0,0,0,0,  # 0-4 nose, eyes
			0,0,        # 5-6 ears
			1,1,        # 7-8 shoulders
			1,1,        # 9-10 elbows
			1,1,        # 11-12 wrists
			0,0,        # 13-14 hips
			0,0,        # 15-16 knees
		], dtype=tf.float32)
		
		# Mask for wrist symmetry: highlight both wrists and elbows
		mask_wrist_symmetry = tf.constant([
			0,0,0,0,0,  # 0-4 nose, eyes
			0,0,        # 5-6 ears
			0,0,        # 7-8 shoulders
			1,1,        # 9-10 elbows
			1,1,        # 11-12 wrists
			0,0,        # 13-14 hips
			0,0,        # 15-16 knees
		], dtype=tf.float32)
		
		masks = tf.stack([mask_elbow_flare, mask_wrist_symmetry], axis=0)
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
		"""Internal build method for OverheadPoseCNN model."""
		model = OverheadPoseCNN()
		
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
	return OverheadPoseCNN._build_model_internal(dropout_rate=dropout_rate, learning_rate=learning_rate)
