# Prompt templates for VLM-assisted fashion captioning

This document provides the prompt templates used for garment captioning and
visual try-on evaluation. The templates are written in a reproducible
open-source format so that each prompt can be copied directly into a VLM API or
included in a paper appendix.

## Usage conventions

Use these conventions when running the prompts.

- `System prompt` defines the role, scope, and high-level task.
- `User prompt` defines the concrete input request and output constraints.
- Bracketed names, such as `[Garment Image]`, `[Ground Truth Image]`, and
  `[Generated Image]`, denote image inputs passed to the VLM.
- Keep decoding deterministic when possible, for example by using temperature
  `0` or the lowest supported value, to reduce prompt-induced variance.
- Do not add extra explanations outside the required output format.

## Prompt 1: Top-garment captioning

This prompt captions the upper-body garment worn by the person in the input
image. It is intended for VLM-assisted caption generation with
`Gemini-2.0-Flash`.

### Prompt card

The prompt card below summarizes the intended use and constraints.

- **Task:** Generate a one-sentence caption for the top garment.
- **Input:** One garment image containing a person wearing an upper-body
  garment.
- **Output:** A single descriptive sentence.
- **Required attributes:** Material, color, sleeve length, and salient garment
  details.
- **Forbidden wording:** Avoid uncertainty markers such as "looks like" and
  "might be."

### System prompt

Use the following text as the system message.

```text
You are a professional fashion analyst. Your task is to describe the top
garment worn by the person in the image in a single, descriptive sentence.
```

### User prompt

Use the following text as the user message.

```text
Analyze the provided garment image. Your description must definitively state
the fabric material, avoiding any words of uncertainty like "looks like" or
"might be". Also include the color, sleeve length, and other key details.

Follow the structure of these examples:

Example 1: A woman is wearing a red cotton-linen short-sleeved shirt.
Example 2: A man is wearing a black silk long-sleeved shirt.
Example 3: A man is wearing a green nylon vest with zippered pockets.
Example 4: A woman is wearing a sleeveless pure cotton tank top.
```

### Expected output examples

The examples below illustrate valid one-sentence captions.

```text
A woman is wearing a red cotton-linen short-sleeved shirt.
A man is wearing a green nylon vest with zippered pockets.
```

## Prompt 2: Bottom-garment captioning

This prompt captions the lower-body garment worn by the person in the input
image. It is intended for VLM-assisted caption generation with
`Gemini-2.0-Flash`.

### Prompt card

The prompt card below summarizes the intended use and constraints.

- **Task:** Generate a one-sentence caption for the pants or skirt.
- **Input:** One garment image containing a person wearing a lower-body
  garment.
- **Output:** A single descriptive sentence.
- **Required attributes:** Material, color, garment category, patterns, text,
  and visible structural features.
- **Forbidden wording:** Avoid uncertainty markers such as "looks like" and
  "might be."

### System prompt

Use the following text as the system message.

```text
You are a professional fashion analyst. Your task is to describe the pants or
skirt worn by the person in the image in a single, descriptive sentence.
```

### User prompt

Use the following text as the user message.

```text
Analyze the provided garment image. Your description must definitively state
the fabric material, avoiding any words of uncertainty like "looks like" or
"might be". Also include the color and other specific details like patterns,
text, or features.

Follow the structure of these examples:

Example 1: A woman is wearing a red linen skirt with a pattern on the left side
and a heart on the right.
Example 2: A man is wearing blue denim jeans with the letters "ABC" printed on
the left leg.
Example 3: A man is wearing black leather pants featuring zippers above the
knees.
Example 4: A woman is wearing white silk trousers with a wide-leg cut.
```

### Expected output examples

The examples below illustrate valid one-sentence captions.

```text
A man is wearing blue denim jeans with the letters "ABC" printed on the left
leg.
A woman is wearing white silk trousers with a wide-leg cut.
```
