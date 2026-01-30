"""
VLM-based Evaluator
Uses Qwen VL API for multi-dimensional image quality evaluation
"""

import base64
import json
import re
import numpy as np
from PIL import Image
from typing import Dict, List
from loguru import logger
from io import BytesIO
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class VLMEvaluator:
    """VLM API-based evaluator"""

    # System prompt
    SYSTEM_PROMPT = """You are a professional Fashion Image Quality Evaluator. You will be conducting a strict "Reference-based Virtual Try-On" assessment.

Input Data:
I will provide you with three images:
1. [Garment Image]: The target clothing (flat lay or mannequin).
2. [Ground Truth Image]: The real photo of the person wearing the clothes (Reference for identity, pose, and background).
3. [Generated Image]: The AI-generated try-on result.

Task:
Compare the [Generated Image] against the references and score it on 5 specific dimensions. Use a scale of 1 to 5 (where 5 is perfect/identical, and 1 is failure)."""

    # User prompt (evaluation criteria)
    EVALUATION_PROMPT = """Please evaluate the [Generated Image] based on the following 5 dimensions. Be critical and look for subtle artifacts.

**Dimension 1: Background Consistency**
Definition: The background (walls, floor, objects, shadows) must remain pixel-perfectly unchanged compared to the [Ground Truth Image], except for the area occluded by the clothing.
Look for: Distortions, warping near the body, color shifts, or blurring in the non-editing areas.
Scoring:
- 5: Background is identical to Ground Truth.
- 4: Minor color/brightness variations, but no structural changes.
- 3: Noticeable but acceptable blurring or slight warping in small areas.
- 2: Obvious artifacts, warping, or color shifts in multiple areas.
- 1: Obvious warping, noise, or structure changes in the background.

**Dimension 2: Person Identity & Body Consistency**
Definition: The model's identity (face), skin tone, and body structure (limbs, hands) must match the [Ground Truth Image].
Look for:
- Face: Is the ID preserved? (No face swapping/distortion).
- Body: Are hands/fingers complete and natural? Are arms the correct thickness?
- Skin: Is the skin tone consistent at the neck/arm boundaries?
Scoring:
- 5: Identity preserved perfectly; hands/limbs are flawless.
- 4: Minor imperfections in hands or body edges, but identity intact.
- 3: Noticeable issues with limbs or slight face distortion, but recognizable.
- 2: Significant body deformations or partial identity loss.
- 1: Face distorted, identity lost, or missing/mutated limbs.

**Dimension 3: Texture Fidelity**
Definition: The fine details from the [Garment Image] (logos, prints, fabric material, complex patterns) must be accurately rendered on the Generated Image.
Look for: Blurred logos, disappeared patterns, or wrong fabric material (e.g., silk looking like cotton).
Scoring:
- 5: Texture/Patterns are sharp and strictly consistent with the Garment Image.
- 4: Minor blur in complex patterns, but overall recognizable.
- 3: Noticeable blur or simplification, but main patterns preserved.
- 2: Significant pattern loss or wrong fabric material appearance.
- 1: Logo missing, pattern completely wrong or blurry.

**Dimension 4: Shape Preservation **
Definition: The geometric style of the clothing from the [Garment Image] must be preserved.
Look for:
- Sleeve Length: Long sleeve vs. Short sleeve vs. Sleeveless.
- Neckline: V-neck vs. Round neck vs. Collar.
- Fit: Baggy vs. Tight.
Scoring:
- 5: The cut and shape perfectly match the specific style of the Garment Image.
- 4: Very minor deviations in fit or cut, but style is correct.
- 3: Noticeable but acceptable deviations (e.g., slightly wrong fit).
- 2: Significant style deviation (e.g., collar shape wrong).
- 1: Wrong category (e.g., long sleeve became short sleeve).

**Dimension 5: Overall Realism **
Definition: Does the [Generated Image] look like a real photograph? Focus on the fusion of the cloth with the person.
Look for:
- Lighting/Shadows: Do shadows on the cloth match the environment?
- Folds: Are the clothing folds natural (gravity-compliant)?
- Artifacts: Any sticker-like edges, white borders, or blurry patches?
Scoring:
- 5: Indistinguishable from a real photo.
- 4: Very realistic, only minor artifacts under close inspection.
- 3: Acceptable realism, some visible artifacts but believable.
- 2: Obvious artifacts, unnatural folds, or lighting mismatches.
- 1: Fake, obvious collage, or severe artifacts.

**Response Format:**
You MUST return a complete JSON with BOTH reasoning and scores. **Keep each reasoning analysis concise (max 100 words).**

Example format (you MUST follow this EXACTLY):
```json
{
  "reasoning": {
    "background_analysis": "Brief analysis (max 100 words)...",
    "person_analysis": "Brief analysis (max 100 words)...",
    "garment_analysis": "Brief analysis (max 100 words)...",
    "realism_analysis": "Brief analysis (max 100 words)..."
  },
  "scores": {
    "background_consistency": 4.5,
    "person_consistency": 4.0,
    "texture_fidelity": 3.5,
    "shape_preservation": 4.0,
    "overall_realism": 4.0
  },
  "final_weighted_score": 4.0
}
```

**CRITICAL - You MUST include:**
1. "reasoning" object with 4 analysis fields (each max 100 words)
2. "scores" object with all 5 numeric scores (1.0-5.0)
3. "final_weighted_score" (1.0-5.0)
4. NO extra text outside the JSON
5. NO missing fields"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen-vl-plus",
        max_tokens: int = 512,
        max_workers: int = 10
    ):
        """
        Initialize VLM evaluator (API mode).

        Args:
            api_key: API key
            base_url: API endpoint URL
            model_name: Model name
            max_tokens: Maximum generation token count
            max_workers: Maximum concurrent API call threads
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_workers = max_workers

        logger.info(f"Initializing VLM Evaluator with API...")
        logger.info(f"  Base URL: {base_url}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Max Workers: {max_workers}")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        logger.success("VLM Evaluator initialized!")

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        # Convert to RGB (remove alpha channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def evaluate_single(
        self,
        source_image: Image.Image,
        ground_truth: Image.Image,
        generated_image: Image.Image
    ) -> Dict:
        """
        Evaluate a single sample.

        Args:
            source_image: Source garment image
            ground_truth: Real target image
            generated_image: Generated image

        Returns:
            Dictionary containing scores for five dimensions
        """
        try:
            # Convert images to base64
            source_b64 = self._image_to_base64(source_image)
            gt_b64 = self._image_to_base64(ground_truth)
            gen_b64 = self._image_to_base64(generated_image)

            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[Garment Image]:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{source_b64}"}
                        },
                        {
                            "type": "text",
                            "text": "[Ground Truth Image]:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"}
                        },
                        {
                            "type": "text",
                            "text": "[Generated Image]:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}"}
                        },
                        {
                            "type": "text",
                            "text": self.EVALUATION_PROMPT
                        }
                    ]
                }
            ]

            # Call API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.0  # Use deterministic output
            )

            # Parse response
            response_text = completion.choices[0].message.content.strip()
            scores = self._parse_response(response_text)

            return scores

        except Exception as e:
            # Check if content moderation failed (inappropriate content)
            error_str = str(e)
            if 'data_inspection_failed' in error_str or 'inappropriate content' in error_str.lower():
                logger.warning(f"Content filter triggered (inappropriate content detected), skipping sample...")
                return None  # Return None to indicate skip

            # Other API errors (network timeout, quota limit, etc.)
            logger.error(f"Error evaluating with VLM API: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._default_scores()

    def evaluate_batch(
        self,
        source_images: List[Image.Image],
        ground_truths: List[Image.Image],
        generated_images: List[Image.Image],
        show_progress: bool = True
    ) -> tuple:
        """
        Batch evaluation (using multi-threaded concurrent processing).

        Args:
            source_images: List of source garment images
            ground_truths: List of real target images
            generated_images: List of generated images
            show_progress: Whether to show progress

        Returns:
            (valid_scores_list, skipped_count) tuple
            Note: Returned list only contains successfully evaluated samples (None results filtered)
        """
        def evaluate_single_wrapper(idx: int, source: Image.Image, gt: Image.Image, generated: Image.Image):
            """
            Thread-safe wrapper for single evaluation.

            Args:
                idx: Sample index
                source: Source garment image
                gt: Real target image
                generated: Generated image

            Returns:
                (idx, scores, error) tuple
            """
            try:
                score = self.evaluate_single(source, gt, generated)
                return (idx, score, None)
            except Exception as e:
                logger.error(f"Sample {idx} evaluation failed: {e}")
                return (idx, None, str(e))

        # Pre-allocate results list
        num_samples = len(source_images)
        all_results = [None] * num_samples  # Save all results (including None)
        skipped_count = 0

        # Use thread pool for concurrent processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {}
            for idx, (source, gt, gen) in enumerate(zip(source_images, ground_truths, generated_images)):
                future = executor.submit(evaluate_single_wrapper, idx, source, gt, gen)
                futures[future] = idx

            # Collect results (with progress bar)
            if show_progress:
                from tqdm import tqdm
                for future in tqdm(as_completed(futures), total=len(futures), desc="VLM Evaluation", unit="image"):
                    idx, score, error = future.result()
                    if error:
                        logger.error(f"Sample {idx} failed: {error}")
                        score = self._default_scores()

                    all_results[idx] = score
                    if score is None:
                        skipped_count += 1
            else:
                for future in as_completed(futures):
                    idx, score, error = future.result()
                    if error:
                        score = self._default_scores()

                    all_results[idx] = score
                    if score is None:
                        skipped_count += 1

        # Filter out None results, only return valid scores
        valid_results = [r for r in all_results if r is not None]

        logger.info(f"VLM batch evaluation complete: {len(valid_results)} valid, {skipped_count} skipped (content filtered)")

        return valid_results, skipped_count

    def _parse_response(self, response: str) -> Dict:
        """
        Parse VLM JSON response.

        Args:
            response: VLM text response

        Returns:
            Parsed scores dictionary
        """
        try:
            # Remove markdown code block markers
            response = response.replace('```json', '').replace('```', '').strip()

            # Try to extract JSON part
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON found in response, using default scores")
                logger.debug(f"Response text: {response[:200]}...")
                return self._default_scores()

            json_str = response[start_idx:end_idx]

            # Clean JSON string: remove trailing commas (common LLM output issue)
            # Remove commas after last item in objects/arrays: }, or ],
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # Try to fix incomplete JSON (missing closing braces)
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                # Add missing closing braces
                json_str += '}' * (open_braces - close_braces)
                logger.debug(f"Added {open_braces - close_braces} missing closing braces")

            parsed = json.loads(json_str)

            # Check if scores field exists
            if 'scores' in parsed and isinstance(parsed['scores'], dict):
                scores_dict = parsed['scores']
                reasoning = parsed.get('reasoning', {})
                final_weighted = parsed.get('final_weighted_score', None)
            elif 'scores' not in parsed and 'reasoning' in parsed:
                # VLM only returned reasoning, no scores
                logger.warning("VLM response missing 'scores' field - using default scores")
                logger.debug(f"Response contained only reasoning, no scores provided")
                # Use default scores
                return self._default_scores_with_reasoning(parsed.get('reasoning', {}))
            else:
                # Backward compatibility for old format (flat structure)
                scores_dict = parsed
                reasoning = parsed.get('reasoning', {})
                final_weighted = None

            # Validate and extract required fields (using new field names)
            # Support multiple possible field name variants
            required_fields = {
                'background_consistency': 'background_consistency',
                'person_consistency': 'person_consistency',
                'texture_fidelity': 'garment_texture',  # New field name -> old field name mapping
                'shape_preservation': 'garment_shape',  # New field name -> old field name mapping
                'overall_realism': 'overall_realism'
            }

            # Additional field name variants (LLM may return different names)
            field_variants = {
                'background_consistency': ['background_consistency', 'background'],
                'person_consistency': ['person_consistency', 'person', 'person_identity'],
                'texture_fidelity': ['texture_fidelity', 'garment_texture', 'texture'],
                'shape_preservation': ['shape_preservation', 'garment_shape', 'shape'],
                'overall_realism': ['overall_realism', 'realism']
            }

            missing_fields = []
            scores = {}

            for new_field, old_field in required_fields.items():
                score = None

                # Try all possible field name variants
                for variant in field_variants.get(new_field, [new_field]):
                    if variant in scores_dict:
                        score = scores_dict[variant]
                        break

                # If still not found, try old field name
                if score is None and old_field in scores_dict:
                    score = scores_dict[old_field]

                if score is None:
                    missing_fields.append(new_field)
                    score = 3.0  # Default medium score

                # Ensure score is between 1-5, support float
                try:
                    scores[old_field] = max(1.0, min(5.0, float(score)))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid score value for {new_field}: {score}, using 3.0")
                    scores[old_field] = 3.0

            # If there are missing fields, log one warning (including all missing fields)
            if missing_fields:
                logger.warning(f"Missing fields in VLM response: {', '.join(missing_fields)}")
                logger.debug(f"Available fields in response: {list(scores_dict.keys())}")
                logger.debug(f"Response preview: {json.dumps(parsed, ensure_ascii=False)[:300]}...")

            # Add reasoning
            scores['reasoning'] = reasoning

            # Add final_weighted_score (if available)
            if final_weighted is not None:
                try:
                    scores['final_weighted_score'] = max(1.0, min(5.0, float(final_weighted)))
                except (ValueError, TypeError):
                    pass

            return scores

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response[:500]}...")

            # Try more aggressive repair strategy: extract score fields
            try:
                scores = self._extract_scores_from_broken_json(response)
                if scores:
                    logger.info("Successfully extracted scores from broken JSON")
                    return scores
            except Exception as extract_error:
                logger.debug(f"Failed to extract scores: {extract_error}")

            return self._default_scores()
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            logger.debug(f"Response text: {response[:500]}...")
            import traceback
            logger.debug(traceback.format_exc())
            return self._default_scores()

    def _extract_scores_from_broken_json(self, response: str) -> Dict:
        """
        Extract scores from incomplete JSON response.

        Args:
            response: Possibly incomplete JSON response

        Returns:
            Extracted scores dictionary, or None if failed
        """
        scores = {}

        # Use regex to extract scores
        patterns = {
            'background_consistency': r'"background_consistency"\s*:\s*([0-9.]+)',
            'person_consistency': r'"person_consistency"\s*:\s*([0-9.]+)',
            'garment_texture': r'"texture_fidelity"\s*:\s*([0-9.]+)',
            'garment_shape': r'"shape_preservation"\s*:\s*([0-9.]+)',
            'overall_realism': r'"overall_realism"\s*:\s*([0-9.]+)'
        }

        for field_name, pattern in patterns.items():
            match = re.search(pattern, response)
            if match:
                try:
                    score = float(match.group(1))
                    scores[field_name] = max(1.0, min(5.0, score))
                except (ValueError, TypeError):
                    scores[field_name] = 3.0
            else:
                scores[field_name] = 3.0

        # Need at least 3 scores extracted to consider success
        valid_scores = [v for v in scores.values() if v != 3.0]
        if len(valid_scores) >= 3:
            scores['reasoning'] = {
                field: 'Extracted from incomplete JSON response'
                for field in scores.keys()
            }
            return scores

        return None

    def _default_scores(self) -> Dict:
        """Return default scores (all 3, indicating medium)"""
        return {
            'background_consistency': 3,
            'person_consistency': 3,
            'garment_texture': 3,
            'garment_shape': 3,
            'overall_realism': 3,
            'reasoning': {
                'background_consistency': 'Error during evaluation',
                'person_consistency': 'Error during evaluation',
                'garment_texture': 'Error during evaluation',
                'garment_shape': 'Error during evaluation',
                'overall_realism': 'Error during evaluation'
            }
        }

    def _default_scores_with_reasoning(self, reasoning: Dict) -> Dict:
        """Return default scores but preserve VLM-provided reasoning"""
        return {
            'background_consistency': 3.0,
            'person_consistency': 3.0,
            'garment_texture': 3.0,
            'garment_shape': 3.0,
            'overall_realism': 3.0,
            'reasoning': reasoning if reasoning else {
                'background_consistency': 'Reasoning provided but scores missing',
                'person_consistency': 'Reasoning provided but scores missing',
                'garment_texture': 'Reasoning provided but scores missing',
                'garment_shape': 'Reasoning provided but scores missing',
                'overall_realism': 'Reasoning provided but scores missing'
            }
        }

    def compute_average_scores(self, scores_list: List[Dict]) -> Dict:
        """
        Compute average scores.

        Args:
            scores_list: List of score results

        Returns:
            Average scores dictionary
        """
        if not scores_list:
            return self._default_scores()

        dimensions = [
            'background_consistency',
            'person_consistency',
            'garment_texture',
            'garment_shape',
            'overall_realism'
        ]

        averages = {}
        for dim in dimensions:
            scores = [s[dim] for s in scores_list if dim in s]
            averages[dim] = np.mean(scores) if scores else 3.0

        # Compute total score
        averages['total_score'] = np.mean(list(averages.values()))

        return averages
