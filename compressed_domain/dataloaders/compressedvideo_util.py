import numpy as np
import torch
from PIL import Image

from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
)

import coviar

# CLIP's RGB statistics.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CompressedVideoExtractor:
    """
    Extract compressed-domain video representations using CoViAR.

    The extractor is responsible ONLY for the video side.

    For every video it can produce three independent temporal sequences:

        iframe      -> one representation per GOP
        residuals   -> selected residuals from each GOP
        mv          -> selected motion vectors from each GOP

    The three sequences can have different lengths.

    Example:

        GOP size = 12
        iframe_sampling_rate = 1
        residual_sampling_rate = 3
        mv_sampling_rate = 3

    For a video containing 5 GOPs:

        iframe:
            5 samples

        residuals:
            approximately 15 samples

        motion vectors:
            approximately 15 samples

    The dataset class is responsible for deciding the final maximum
    sequence lengths and padding/masking them.
    """

    # CoViAR representation IDs.
    IFRAME = 0
    MV = 1
    RESIDUAL = 2

    def __init__(
        self,
        gop_size=12,
        image_resolution=224,
        iframe_sampling_rate=1,
        residual_sampling_rate=3,
        mv_sampling_rate=3,
        accumulate=True,
        max_gops=None,
        gop_slice=2,
        bgr_to_rgb=True,
        normalize=True,
        mv_clip=20.0,
    ):
        """
        Parameters
        ----------
        gop_size : int
            Number of frame positions assumed per GOP.

            For the standard CoViAR setup this is 12.

        image_resolution : int
            Spatial size expected by the downstream vision encoder.

        iframe_sampling_rate : int
            Number of I-frames sampled from each GOP.

            This should remain 1 because an I-frame representation
            exists only at the beginning of a GOP.

        residual_sampling_rate : int
            Number of residual representations sampled from each GOP.

            Example:
                3 -> select 3 residuals from every GOP.

        mv_sampling_rate : int
            Number of motion-vector representations sampled from each GOP.

            Example:
                3 -> select 3 MVs from every GOP.

        accumulate : bool
            Passed directly to coviar.load().

            True:
                returns accumulated representation.

            False:
                returns the original compressed representation.

        max_gops : int or None
            Temporal budget: at most this many GOPs are decoded per video,
            and all three modalities are taken from the SAME GOPs, so they
            always describe the same time span. None decodes every GOP.

        gop_slice : int
            Which GOPs are kept when a video has more than max_gops:
            0 = first ones, 1 = last ones, 2 = uniformly spread.

        bgr_to_rgb : bool
            CoViAR returns I-frames and residuals in BGR channel order.
            CLIP (and the CLIP-initialised residual encoder) expects RGB.

        normalize : bool
            Scale residuals and motion vectors to the same order of
            magnitude as CLIP-normalised I-frames (see _preprocess_*).

        mv_clip : float
            Motion vectors are clamped to +-mv_clip pixels and divided by
            it (the same +-20 px window CoViAR uses).
        """

        if iframe_sampling_rate != 1:
            raise ValueError(
                "iframe_sampling_rate must be 1 because each GOP "
                "contains only one I-frame representation."
            )

        if gop_size <= 0:
            raise ValueError("gop_size must be positive.")

        if residual_sampling_rate < 0:
            raise ValueError("residual_sampling_rate must be >= 0.")

        if mv_sampling_rate < 0:
            raise ValueError("mv_sampling_rate must be >= 0.")

        self.gop_size = gop_size
        self.image_resolution = image_resolution

        self.iframe_sampling_rate = iframe_sampling_rate
        self.residual_sampling_rate = residual_sampling_rate
        self.mv_sampling_rate = mv_sampling_rate

        self.accumulate = accumulate

        if max_gops is not None and max_gops <= 0:
            raise ValueError("max_gops must be positive or None.")

        if gop_slice not in (0, 1, 2):
            raise ValueError("gop_slice must be 0, 1 or 2.")

        self.max_gops = max_gops
        self.gop_slice = gop_slice
        self.bgr_to_rgb = bgr_to_rgb
        self.normalize = normalize
        self.mv_clip = float(mv_clip)

        # Residuals are zero-centred differences of 0..255 pixels, so they
        # are only divided by the scale CLIP applies to pixels (255 * std);
        # no mean is subtracted.
        self.residual_scale = 255.0 * torch.tensor(
            CLIP_STD, dtype=torch.float32
        ).view(3, 1, 1)

        # ---------------------------------------------------------------
        # I-frame preprocessing
        #
        # I-frames are ordinary RGB images and can therefore use the
        # same kind of preprocessing normally used by CLIP-style models.
        # ---------------------------------------------------------------
        self.iframe_transform = self._build_iframe_transform(
            image_resolution
        )

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _build_iframe_transform(self, size):
        """
        Build preprocessing for I-frames.

        CoViAR returns the I-frame as an image-like NumPy array.
        We convert it to PIL and apply the usual:

            resize -> center crop -> RGB -> tensor -> normalize

        pipeline.

        These normalization values are CLIP's standard RGB statistics.
        """

        return Compose([
            Resize(size),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def _resize_and_crop_array(self, array):
        """
        Resize/crop a CoViAR representation while preserving channels.

        This function is intentionally kept separate from the I-frame
        transform because residuals and motion vectors are NOT ordinary
        RGB images.

        Parameters
        ----------
        array : np.ndarray
            H x W x C representation.

            I-frame/residual:
                C = 3

            motion vector:
                C = 2

        Returns
        -------
        np.ndarray
            H' x W' x C
        """

        h, w = array.shape[:2]

        # Resize while preserving aspect ratio.
        scale = self.image_resolution / min(h, w)

        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resized = np.empty(
            (new_h, new_w, array.shape[2]),
            dtype=array.dtype,
        )

        # Resize each channel independently.
        #
        # We deliberately avoid treating an MV as RGB because its
        # two channels represent motion-vector components rather than
        # colors.
        for c in range(array.shape[2]):
            channel = Image.fromarray(array[..., c])
            channel = channel.resize(
                (new_w, new_h),
                Image.BICUBIC,
            )
            resized[..., c] = np.asarray(channel)

        # Center crop.
        top = (new_h - self.image_resolution) // 2
        left = (new_w - self.image_resolution) // 2

        cropped = resized[
            top:top + self.image_resolution,
            left:left + self.image_resolution,
        ]

        return cropped

    # ------------------------------------------------------------------
    # GOP bookkeeping
    # ------------------------------------------------------------------

    def get_num_gops(self, video_path):
        """
        Return the number of GOPs in a video.

        CoViAR exposes the total number of frames through
        get_num_frames(). We divide those frames into fixed-size GOPs.

        Example:

            total frames = 25
            GOP size     = 12

            GOP 0 -> frames 0..11
            GOP 1 -> frames 12..23
            GOP 2 -> frame 24

            number of GOPs = 3
        """

        num_frames = coviar.get_num_frames(video_path)

        if num_frames <= 0:
            return 0

        return int(np.ceil(num_frames / self.gop_size))

    def _get_gop_length(self, num_frames, gop_index):
        """
        Return the number of actual frames in one GOP.

        This matters for the final GOP, which may contain fewer than
        gop_size frames.

        Example:

            total frames = 25
            gop_size = 12

            GOP 0 -> 12 frames
            GOP 1 -> 12 frames
            GOP 2 -> 1 frame
        """

        start_frame = gop_index * self.gop_size

        remaining = num_frames - start_frame

        return max(0, min(self.gop_size, remaining))

    # ------------------------------------------------------------------
    # Sampling positions
    # ------------------------------------------------------------------

    def _sample_iframe_positions(self, gop_length):
        """
        Return frame positions used for I-frame sampling.

        There is exactly one I-frame per GOP:

            frame_index = 0

        If the GOP is empty, return no samples.
        """

        if gop_length <= 0:
            return []

        return [0]

    def _sample_delta_positions(self, gop_length, sampling_rate):
        """
        Select residual/MV frame positions inside one GOP.

        Position 0 is the I-frame and therefore MUST NOT be selected.

        For example, with:

            GOP size = 12
            sampling_rate = 3

        the available compressed-domain delta positions are:

            1, 2, 3, ..., 11

        and we select 3 positions approximately uniformly across them.

        We use linspace rather than simply taking [1, 2, 3],
        because [1, 2, 3] would heavily bias sampling toward the
        beginning of every GOP.

        If the final GOP is shorter, only its actually available
        delta frames are considered.
        """

        # A GOP containing only the I-frame has no residual/MV samples.
        if gop_length <= 1 or sampling_rate <= 0:
            return []

        available_positions = np.arange(
            1,
            gop_length,
            dtype=np.int64,
        )

        # If fewer positions exist than requested, use all of them.
        num_samples = min(
            sampling_rate,
            len(available_positions),
        )

        # A single sample is taken from the middle of the GOP: the first
        # P-frame carries almost no accumulated motion/residual.
        if num_samples == 1:
            return [int(available_positions[len(available_positions) // 2])]

        # Select approximately uniformly spaced positions.
        indices = np.linspace(
            0,
            len(available_positions) - 1,
            num=num_samples,
            dtype=np.int64,
        )

        return available_positions[indices].tolist()

    def _select_gops(self, num_gops):
        """
        Return the GOP indices decoded for a video with num_gops GOPs.

        All modalities are sampled from these same GOPs, which keeps
        them temporally aligned under a shared budget (max_gops).
        """

        if self.max_gops is None or num_gops <= self.max_gops:
            return list(range(num_gops))

        if self.gop_slice == 0:
            return list(range(self.max_gops))

        if self.gop_slice == 1:
            return list(range(num_gops - self.max_gops, num_gops))

        return np.linspace(
            0,
            num_gops - 1,
            num=self.max_gops,
            dtype=np.int64,
        ).tolist()

    # ------------------------------------------------------------------
    # Individual CoViAR loading
    # ------------------------------------------------------------------

    def _load_representation(
        self,
        video_path,
        gop_index,
        frame_index,
        representation_type,
    ):
        """
        Load one compressed representation from CoViAR.

        Parameters
        ----------
        video_path : str
            Path to .mp4 file.

        gop_index : int
            GOP index.

        frame_index : int
            Frame position INSIDE that GOP.

        representation_type : int
            0 = I-frame
            1 = motion vector
            2 = residual
        """

        representation = coviar.load(
            video_path,
            gop_index,
            frame_index,
            representation_type,
            self.accumulate,
        )

        return representation

    # ------------------------------------------------------------------
    # Representation preprocessing
    # ------------------------------------------------------------------

    def _preprocess_iframe(self, iframe):
        """
        Convert one CoViAR I-frame into a model-ready tensor.

        CoViAR's image representation is handled as an RGB image.

        Returns:
            [3, H, W]
        """

        if iframe is None:
            return None

        # CoViAR's output is image-like, in BGR channel order.
        if self.bgr_to_rgb:
            iframe = np.ascontiguousarray(iframe[..., ::-1])

        image = Image.fromarray(iframe)

        return self.iframe_transform(image)

    def _preprocess_residual(self, residual):
        """
        Convert one residual representation to a tensor.

        Residuals are signed, zero-centred pixel differences. They are
        put in the I-frames' channel order and divided by the scale CLIP
        applies to pixels (255 * std), so the CLIP-initialised residual
        encoder sees inputs of the magnitude it was pretrained on. No
        mean is subtracted.

        Returns:
            [3, H, W]
        """

        if residual is None:
            return None

        # CoViAR residuals are signed differences.
        #
        # We keep their values instead of treating them as ordinary
        # uint8 RGB pixels.
        residual = residual.astype(np.float32)

        # Same channel order as the I-frames.
        if self.bgr_to_rgb:
            residual = np.ascontiguousarray(residual[..., ::-1])

        residual = self._resize_and_crop_array(residual)

        residual = torch.from_numpy(
            np.ascontiguousarray(residual.transpose(2, 0, 1))
        ).float()

        if self.normalize:
            residual = residual / self.residual_scale

        return residual

    def _preprocess_mv(self, mv):
        """
        Convert one motion-vector representation to a tensor.

        Motion vectors normally have two channels:

            channel 0 -> horizontal displacement
            channel 1 -> vertical displacement

        Therefore they must NOT be passed through an RGB image
        preprocessing pipeline.

        Returns:
            [2, H, W]
        """

        if mv is None:
            return None

        mv = mv.astype(np.float32)

        # Clamp to +-mv_clip pixels and scale to [-1, 1] BEFORE resizing,
        # so a few huge accumulated vectors cannot dominate the bicubic
        # interpolation.
        if self.normalize:
            mv = np.clip(mv, -self.mv_clip, self.mv_clip) / self.mv_clip

        mv = self._resize_and_crop_array(mv)

        mv = torch.from_numpy(
            np.ascontiguousarray(mv.transpose(2, 0, 1))
        ).float()

        if self.normalize:
            # Bicubic interpolation can overshoot slightly.
            mv = mv.clamp_(-1.0, 1.0)

        return mv

    # ------------------------------------------------------------------
    # Main extraction function
    # ------------------------------------------------------------------

    def video_to_compressed_tensors(self, video_path):
        """
        Extract all three compressed-domain modalities from one video.

        Returns
        -------
        dict with:

            {
                "iframe":    Tensor [N_I, 3, H, W],
                "residuals": Tensor [N_R, 3, H, W],
                "mv":        Tensor [N_M, 2, H, W],
            }

        The lengths N_I, N_R and N_M are allowed to differ.

        Importantly, this function DOES NOT pad them.

        Padding is the dataset-level responsibility because each
        modality can have its own max_length.
        """

        num_frames = coviar.get_num_frames(video_path)

        if num_frames <= 0:
            return {
                "iframe": torch.empty(
                    0, 3,
                    self.image_resolution,
                    self.image_resolution,
                ),
                "residuals": torch.empty(
                    0, 3,
                    self.image_resolution,
                    self.image_resolution,
                ),
                "mv": torch.empty(
                    0, 2,
                    self.image_resolution,
                    self.image_resolution,
                ),
            }

        num_gops = self.get_num_gops(video_path)

        iframe_samples = []
        residual_samples = []
        mv_samples = []

        # ---------------------------------------------------------------
        # Iterate through GOPs, NOT seconds.
        #
        # This is the fundamental difference from RawVideoExtractor.
        # ---------------------------------------------------------------
        for gop_index in self._select_gops(num_gops):

            gop_length = self._get_gop_length(
                num_frames,
                gop_index,
            )

            # -----------------------------------------------------------
            # I-frame
            #
            # There is one I-frame at position 0 of each GOP.
            # -----------------------------------------------------------
            iframe_positions = self._sample_iframe_positions(
                gop_length
            )

            for frame_index in iframe_positions:

                iframe = self._load_representation(
                    video_path,
                    gop_index,
                    frame_index,
                    self.IFRAME,
                )

                iframe = self._preprocess_iframe(iframe)

                if iframe is not None:
                    iframe_samples.append(iframe)

            # -----------------------------------------------------------
            # Residuals
            #
            # Position 0 is excluded because it is the I-frame.
            # -----------------------------------------------------------
            residual_positions = self._sample_delta_positions(
                gop_length,
                self.residual_sampling_rate,
            )

            for frame_index in residual_positions:

                residual = self._load_representation(
                    video_path,
                    gop_index,
                    frame_index,
                    self.RESIDUAL,
                )

                residual = self._preprocess_residual(residual)

                if residual is not None:
                    residual_samples.append(residual)

            # -----------------------------------------------------------
            # Motion vectors
            # -----------------------------------------------------------
            mv_positions = self._sample_delta_positions(
                gop_length,
                self.mv_sampling_rate,
            )

            for frame_index in mv_positions:

                mv = self._load_representation(
                    video_path,
                    gop_index,
                    frame_index,
                    self.MV,
                )

                mv = self._preprocess_mv(mv)

                if mv is not None:
                    mv_samples.append(mv)

        # ---------------------------------------------------------------
        # Stack each modality independently.
        #
        # Empty videos / failed extraction are represented by an empty
        # tensor rather than a strange shape such as [1].
        # ---------------------------------------------------------------
        iframe = self._stack_or_empty(
            iframe_samples,
            channels=3,
        )

        residuals = self._stack_or_empty(
            residual_samples,
            channels=3,
        )

        mv = self._stack_or_empty(
            mv_samples,
            channels=2,
        )

        return {
            "iframe": iframe,
            "residuals": residuals,
            "mv": mv,
        }

    def _stack_or_empty(self, samples, channels):
        """
        Stack a list of [C,H,W] tensors into [N,C,H,W].

        If no samples exist, return:

            [0, C, H, W]

        instead of using a special error shape.
        """

        if len(samples) == 0:
            return torch.empty(
                0,
                channels,
                self.image_resolution,
                self.image_resolution,
            )

        return torch.stack(samples)