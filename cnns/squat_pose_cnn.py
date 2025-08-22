import tensorflow as tf
from tensorflow.keras import layers

# Instruction decoding for external use
INSTRUCTION_MAP = {
    0: "good_form",
    1: "knee_forward_over_ankle",
    2: "squat_not_deep_enough",
}


class SquatPoseCNN(tf.keras.Model):
	"""
	Squat-focused 1D CNN with engineered biomechanical features and rule-based outputs.
	Optimized for sideways camera views where the hidden leg cannot be reliably observed.
	
	Depth Rule: Uses hip-knee-ankle angle measurement with tolerance-based thresholds:
	- standing_threshold: 150° (ignores standing positions safely)
	- proper_squat_threshold: 130° (with ±10° tolerance for wiggle room)
	- Rule only triggers when person is squatting but not deep enough (120°-160° range)
	- Angle-based measurement provides better biomechanical assessment than Y-coordinate differences
	
	Knee-Forward Rule: Single-unit analysis using confidence-based side selection:
	- Selects the knee-ankle pair with higher confidence score from Movenet for measurement
	- Computes knee_forward_offset = knee_x - ankle_x for the selected side only
	- Triggers if offset exceeds knee_forward_threshold + tolerance (5% wiggle room)
	- Benefits: robust to occlusion, sideways views, prevents false positives from hidden legs
	- Visual feedback highlights ONLY the knee and ankle of the higher-confidence side
	
	Features: Focuses on robust, visible-side measurements only:
	- knee_forward_offset: Single-unit forward offset using higher-confidence side only
	- squat_angle_cosine: Single-unit hip-knee-ankle angle cosine for depth assessment
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
		knee_forward_threshold: float = 0.25,  # knee should not be more than this much forward of ankle
		knee_forward_tolerance: float = 0.05,  # tolerance for knee-forward check
		depth_tolerance_degrees: float = 10.0,  # tolerance for depth angle check in degrees
		**kwargs,
	):
		super(SquatPoseCNN, self).__init__(**kwargs)
		self.num_joints = num_joints
		self.knee_forward_threshold = knee_forward_threshold
		self.knee_forward_tolerance = knee_forward_tolerance
		self.depth_tolerance_degrees = depth_tolerance_degrees
		self.reshape_layer = layers.Reshape((num_joints, 3))

		# Precompute depth thresholds using TensorFlow operations
		# cos(120°) ≈ -0.5, cos(150°) ≈ -0.94
		self.proper_squat_cos_threshold = tf.cos(tf.radians(120.0 + depth_tolerance_degrees))  # more lenient
		self.standing_cos_threshold = tf.cos(tf.radians(150.0 - depth_tolerance_degrees))  # more lenient

		# Preprocessing joint mask: keep shoulders, hips, knees, ankles; ignore head, elbows, wrists
		preproc_mask = [
			0, 0, 0,  # nose, eyes
			0, 0,     # ears
			1, 1,     # shoulders
			0, 0,     # elbows
			0, 0,     # wrists
			1, 1,     # hips
			1, 1,     # knees
			1, 1,     # ankles
		]
		self.preprocess_joint_mask = tf.constant(preproc_mask, dtype=tf.float32)
		# Precompute broadcasted joint mask for preprocessing: (1, J, 1)
		self._preprocess_joint_mask_reshaped = tf.reshape(self.preprocess_joint_mask, (1, self.num_joints, 1))

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

		# Engineered features embedding
		# Input: 2 features [knee_forward_offset, squat_angle_cosine], Output: 32 features
		self.biomech_dense = layers.Dense(32, activation='relu', name="biomech_embedding")

		# Fusion + head for learned form score
		self.fusion_dense1 = layers.Dense(64, activation='relu')
		self.fusion_dropout1 = layers.Dropout(0.3)
		self.fusion_dense2 = layers.Dense(32, activation='relu')
		self.fusion_dropout2 = layers.Dropout(0.2)
		self.form_score_output = layers.Dense(1, activation='sigmoid', name='form_score')

		# Precreate rule mask constants to avoid reallocation per call
		self._mask_knee_forward = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], dtype=tf.float32)
		self._mask_depth = tf.constant([0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0], dtype=tf.float32)

	def preprocess_keypoints(self, keypoints: tf.Tensor) -> tf.Tensor:
		"""
		TFLite-optimized preprocessing:
		- Center by mid-hip
		- Scale by torso height
		- Apply preprocessing joint mask efficiently
		Input: (batch, 17, 3) → Output: (batch, 17, 3)
		"""
		xy = keypoints[:, :, :2]           # (batch, 17, 2)
		conf = keypoints[:, :, 2:3]        # (batch, 17, 1)

		# Precomputed mask
		mask = self._preprocess_joint_mask_reshaped  # (1, 17, 1)
		
		# Compute mid-hip and mid-shoulder using **mask-aware sum** and divide by count
		hip_slice = xy[:, 11:13, :] * mask[:, 11:13, :]        # (batch, 2, 2)
		shoulder_slice = xy[:, 5:7, :] * mask[:, 5:7, :]       # (batch, 2, 2)

		# Instead of reduce_mean, sum / count avoids unnecessary division by masked zeros
		mid_hip = tf.reduce_sum(hip_slice, axis=1, keepdims=True) / 2.0
		mid_shoulder = tf.reduce_sum(shoulder_slice, axis=1, keepdims=True) / 2.0

		# Center and scale
		centered = xy - mid_hip                   # (batch, 17, 2)
		torso_vec = mid_shoulder - mid_hip        # (batch, 1, 2)
		torso_height = tf.norm(torso_vec, axis=2, keepdims=True)
		torso_height = tf.maximum(torso_height, 1e-6)
		scaled = centered / torso_height

		# Apply joint mask **once** to scaled coordinates and confidence
		processed = tf.concat([scaled, conf], axis=2) * mask

		return processed

	def _raw_engineered_features(self, keypoints: tf.Tensor) -> tf.Tensor:
		"""
		Compute and normalize squat-relevant features for sideways analysis.
		Returns tensor (batch, 2) with features:
		[knee_forward_offset_normalized, squat_angle_normalized]
		- knee_forward_offset_normalized in [0,1], 0 small/none, 1 at/above threshold
		- squat_angle_normalized in [0,1], 0 shallowest (worst), 1 deepest (best)
		All operations are vectorized and TFLite-friendly.
		"""
		yx = keypoints[:, :, :2]
		conf = keypoints[:, :, 2]  # (batch, 17)

		# Single-unit selection using higher-confidence knee
		knee_conf = tf.stack([conf[:, 13], conf[:, 14]], axis=1)  # (batch, 2)
		is_left = tf.cast(knee_conf[:, 0] > knee_conf[:, 1], tf.float32)  # (batch,)
		is_right = 1.0 - is_left

		# 1) Knee-forward offset normalization
		# MoveNet outputs [y, x, conf], so yx[:, :, 1] gives x-coordinates
		knee_x = yx[:, [13, 14], 1]        # (batch, 2)
		ankle_x = yx[:, [15, 16], 1]       # (batch, 2)
		offsets = knee_x - ankle_x          # (batch, 2)
		knee_forward_offset = offsets[:, 0] * is_left + offsets[:, 1] * is_right  # (batch,)
		knee_forward_offset = tf.abs(knee_forward_offset)
		knee_thresh = tf.convert_to_tensor(self.knee_forward_threshold, dtype=knee_forward_offset.dtype)
		knee_thresh = tf.maximum(knee_thresh, tf.constant(1e-6, dtype=knee_forward_offset.dtype))
		knee_forward_norm = knee_forward_offset / knee_thresh
		knee_forward_norm = tf.clip_by_value(knee_forward_norm, 0.0, 1.0)

		# 2) Squat angle cosine normalization (deep squat => smaller cosine)
		# Select same-side joints consistently
		hip = (yx[:, 11, :] * is_left[:, None] + yx[:, 12, :] * is_right[:, None])   # (batch, 2)
		knee = (yx[:, 13, :] * is_left[:, None] + yx[:, 14, :] * is_right[:, None])  # (batch, 2)
		ankle = (yx[:, 15, :] * is_left[:, None] + yx[:, 16, :] * is_right[:, None]) # (batch, 2)
		squat_cos = self._joint_cos_angle(hip, knee, ankle)  # (batch,)

		# Clip to [standing_cos_threshold (shallow/worst), proper_squat_cos_threshold (deep/best)]
		standing_cos = tf.cast(self.standing_cos_threshold, dtype=squat_cos.dtype)
		proper_cos = tf.cast(self.proper_squat_cos_threshold, dtype=squat_cos.dtype)
		cos_clipped = tf.clip_by_value(squat_cos, standing_cos, proper_cos)
		range_denom = tf.maximum(proper_cos - standing_cos, tf.constant(1e-6, dtype=squat_cos.dtype))
		# Map standing -> 0, proper -> 1
		squat_angle_norm = (cos_clipped - standing_cos) / range_denom
		squat_angle_norm = tf.clip_by_value(squat_angle_norm, 0.0, 1.0)

		# Return normalized features: (batch, 2)
		return tf.stack([knee_forward_norm, squat_angle_norm], axis=1)

	def compute_engineered_features(self, keypoints: tf.Tensor) -> tf.Tensor:
		# _raw_engineered_features returns normalized features (batch, 2)
		normalized_feats = self._raw_engineered_features(keypoints)
		return self.biomech_dense(normalized_feats)
	
	@staticmethod
	def _joint_cos_angle(a: tf.Tensor, b: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
		ba = a - b
		bc = c - b
		ba = tf.math.l2_normalize(ba, axis=-1)
		bc = tf.math.l2_normalize(bc, axis=-1)
		return tf.reduce_sum(ba * bc, axis=-1)

	def _rule_outputs(self, keypoints: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		"""
		TFLite-optimized vectorized rule-based outputs for squat instructions.
		Returns:
		- instruction_id: (batch,) int32
		- joint_mask: (batch, 17) float32 in {0,1}
		"""
		yx = keypoints[:, :, :2]  # (batch, 17, 2)
		conf = keypoints[:, :, 2]  # confidence scores

		# 1. Select side once based on higher knee confidence
		left_knee_conf = conf[:, 13]
		right_knee_conf = conf[:, 14]
		is_left = tf.cast(left_knee_conf > right_knee_conf, tf.float32)  # (batch,)
		is_right = 1.0 - is_left

		# Helper to select same-side joint consistently using multiplication (TFLite-friendly)
		def pick(side_left_idx: int, side_right_idx: int) -> tf.Tensor:
			return (yx[:, side_left_idx, :] * is_left[:, None] + 
					yx[:, side_right_idx, :] * is_right[:, None])

		# Same-side triplet (hip, knee, ankle)
		hip = pick(11, 12)     # (batch, 2) [y, x]
		knee = pick(13, 14)    # (batch, 2)
		ankle = pick(15, 16)   # (batch, 2)

		# 2. Knee-forward check (x-coordinates, with tolerance)
		knee_x = knee[:, 1]    # x-coordinates
		ankle_x = ankle[:, 1]
		knee_forward_offset = tf.abs(knee_x - ankle_x)
		cond_knee_forward = knee_forward_offset > (self.knee_forward_threshold + self.knee_forward_tolerance)

		# 3. Squat depth check using interior vs exterior angle (cosine-based for performance)
		# Use the selected same-side hip-knee-ankle triplet but compute both possible angles
		
		# Compute cosine of angle hip->knee->ankle (interior angle)
		interior_cos = self._joint_cos_angle(hip, knee, ankle)  # (batch,)
		
		# Compute cosine of angle ankle->knee->hip (exterior angle complement)
		exterior_cos = self._joint_cos_angle(ankle, knee, hip)  # (batch,)
		
		# The actual inner knee angle has the LARGER cosine value (since cos is decreasing)
		# cos(120°) ≈ -0.5, cos(240°) ≈ -0.5, but we want the smaller actual angle
		squat_cos = tf.maximum(interior_cos, exterior_cos)  # (batch,)
		
		# Use precomputed thresholds from __init__
		is_squatting = squat_cos < self.standing_cos_threshold
		not_deep_enough = squat_cos > self.proper_squat_cos_threshold
		cond_depth = tf.logical_and(is_squatting, not_deep_enough)

		# 4. Instruction ID (reverted to original logic)
		cond_stack = tf.stack([cond_knee_forward, cond_depth], axis=1)
		cond_stack_i = tf.cast(cond_stack, tf.int32)
		any_true = tf.reduce_any(cond_stack, axis=1)
		max_idx = tf.argmax(cond_stack_i, axis=1)
		instruction_id = tf.where(any_true, max_idx + 1, tf.zeros_like(max_idx))  # 0 = good form

		# 5. Joint mask (reverted to original logic)
		joint_mask = (
			tf.cast(cond_knee_forward, tf.float32)[:, None] * self._mask_knee_forward +
			tf.cast(cond_depth, tf.float32)[:, None] * self._mask_depth
		)
		joint_mask = tf.clip_by_value(joint_mask, 0.0, 1.0)

		return instruction_id, joint_mask

	# Remove for production
	def get_threshold_info(self) -> dict:
		"""
		Get information about the current thresholds for debugging/understanding.
		Returns dict with threshold values and explanations.
		"""
		# Convert TensorFlow tensors to Python values safely
		try:
			proper_squat_cos_val = float(self.proper_squat_cos_threshold)
			standing_cos_val = float(self.standing_cos_threshold)
		except:
			# Fallback for graph mode - use string representation
			proper_squat_cos_val = str(self.proper_squat_cos_threshold)
			standing_cos_val = str(self.standing_cos_threshold)
		
		return {
			"knee_forward_threshold": self.knee_forward_threshold,
			"knee_forward_tolerance": self.knee_forward_tolerance,
			"depth_tolerance_degrees": self.depth_tolerance_degrees,
			"precomputed_thresholds": {
				"proper_squat_cos_threshold": proper_squat_cos_val,
				"standing_cos_threshold": standing_cos_val
			},
			"explanation": {
				"knee_forward_threshold": "Knee should not be more than this much forward of ankle (positive value)",
				"knee_forward_tolerance": f"±{self.knee_forward_tolerance:.3f} tolerance for knee-forward check",
				"depth_measurement": "Uses hip-knee-ankle angle with tolerance-based thresholds",
				"depth_thresholds": {
					"standing_threshold": f"150° - {self.depth_tolerance_degrees}° = {150.0 - self.depth_tolerance_degrees:.1f}° (ignores standing positions)",
					"proper_squat_threshold": f"120° + {self.depth_tolerance_degrees}° = {120.0 + self.depth_tolerance_degrees:.1f}° (with tolerance)",
					"note": "Angle-based measurement provides better biomechanical assessment than Y-coordinate differences"
				},
				"tolerance": f"±{self.knee_forward_tolerance:.3f} tolerance for knee-forward, ±{self.depth_tolerance_degrees}° for depth angles"
			}
		}


	def call(self, inputs: tf.Tensor, training=None):
		# Input validation
		if inputs.shape[-1] != 51:  # 17 joints * 3 (y, x, conf)
			raise ValueError(f"Expected input shape (..., 51), got {inputs.shape}")
		
		x = self.reshape_layer(inputs)
		x_proc = self.preprocess_keypoints(x)

		c = self.conv1(x_proc); c = self.bn1(c, training=training); c = self.relu1(c)
		c = self.conv2(c); c = self.bn2(c, training=training); c = self.relu2(c)
		c = self.conv3(c); c = self.bn3(c, training=training); c = self.relu3(c)
		c = self.global_pool(c)
		cnn_embedding = self.cnn_dense(c)

		biomech_embedding = self.compute_engineered_features(x_proc)

		# Fusion: CNN embedding (32) + Biomechanical features (32) = 64 total features
		# Note: Biomechanical features process 2 input features [knee_forward_offset, squat_angle_cosine] and output 32 features
		# Both features use single-unit selection for robustness to occlusion
		fused = layers.Concatenate()([cnn_embedding, biomech_embedding])
		fused = self.fusion_dense1(fused); fused = self.fusion_dropout1(fused, training=training)
		fused = self.fusion_dense2(fused); fused = self.fusion_dropout2(fused, training=training)

		form_score = self.form_score_output(fused)
		instruction_id, joint_mask = self._rule_outputs(x_proc)

		joint_masked_keypoints = tf.concat([x[:, :, :2], tf.expand_dims(joint_mask, axis=-1)], axis=-1)

		return {
			"form_score": form_score,
			"instruction_id": instruction_id,
			"joint_masked_keypoints": joint_masked_keypoints,
		}

	# Remove for production
	def get_model_info(self) -> dict:
		"""
		Get comprehensive model information including architecture details.
		Useful for debugging and understanding the model structure.
		"""
		return {
			"model_type": "SquatPoseCNN",
			"input_shape": f"(batch, {self.num_joints * 3})",  # 51 for 17 joints
			"architecture": {
				"cnn_backbone": {
					"conv1": "Conv1D(64, 3, padding='same')",
					"conv2": "Conv1D(64, 5, padding='same')", 
					"conv3": "Conv1D(128, 3, padding='same')",
					"global_pool": "GlobalAveragePooling1D",
					"cnn_embedding": "Dense(32)"
				},
				"biomechanical_features": {
					"input_features": 2,  # [knee_forward_offset, squat_angle_cosine]
					"output_features": 32,
					"layer": "Dense(32, activation='relu')"
				},
				"fusion": {
					"input_features": 64,  # 32 + 32
					"fusion_dense1": "Dense(64, activation='relu')",
					"fusion_dropout1": f"Dropout({self.fusion_dropout1.rate})",
					"fusion_dense2": "Dense(32, activation='relu')",
					"fusion_dropout2": f"Dropout({self.fusion_dropout2.rate})",
					"form_score": "Dense(1, activation='sigmoid')"
				}
			},
			"outputs": {
				"form_score": "(batch, 1) - sigmoid in [0,1]",
				"instruction_id": "(batch,) - int32 instruction ID",
				"joint_masked_keypoints": "(batch, 17, 3) - keypoints with mask"
			},
			"thresholds": self.get_threshold_info()
		}

	@staticmethod
	def build_model(dropout_rate: float = 0.3, learning_rate: float = 0.001) -> tf.keras.Model:
		"""
		Build and compile a SquatPoseCNN model for training.
		
		This function allows external scripts (like train.py) to dynamically import 
		and build the model without modifying internal engineered features or rule-based outputs.
		
		Args:
			dropout_rate (float): Dropout rate for regularization (default: 0.3)
			learning_rate (float): Learning rate for Adam optimizer (default: 0.001)
		
		Returns:
			tf.keras.Model: Compiled model ready for training
		
		Note:
			- Only the 'form_score' output is trained
			- Rule-based outputs (instruction_id, joint_masked_keypoints) are preserved
			- Engineered features computation remains unchanged
		"""
		# Instantiate the model with default parameters
		model = SquatPoseCNN()
		
		# Update dropout rates to use the parameter
		model.fusion_dropout1.rate = dropout_rate
		model.fusion_dropout2.rate = dropout_rate * 0.67  # Second dropout slightly lower
		
		# Configure multi-output training
		# Only train the 'form_score' output, preserve rule-based outputs
		losses = {
			'form_score': 'binary_crossentropy',  # Trainable output
			'instruction_id': 'mse',              # Rule-based output (minimal impact)
			'joint_masked_keypoints': 'mse'       # Rule-based output (minimal impact)
		}
		
		# Set loss weights to focus training on form_score
		loss_weights = {
			'form_score': 1.0,                    # Full training weight
			'instruction_id': 0.0,                # No training impact
			'joint_masked_keypoints': 0.0         # No training impact
		}
		
		# Configure metrics for the trainable output
		metrics = {
			'form_score': ['accuracy']
		}
		
		# Compile the model
		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
			loss=losses,
			loss_weights=loss_weights,
			metrics=metrics
		)
		
		return model
